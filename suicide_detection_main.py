import threading
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import joblib
import os
import sys
import time
from datasets import load_dataset
from TermLoading import TermLoading
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback

# Exception handling decorator
def log_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb = traceback.format_tb(exc_tb)
            print(f"Exception '{e}' occurred in function '{func.__name__}' at line {exc_tb.tb_lineno}")
            print(''.join(tb))
    return wrapper

def load_necessary_lib():
    """Load the required NLTK data."""
    nltk.download('stopwords')
    nltk.download('punkt')
    print("Libraries loaded successfully")

def load_dataset_download(dataset_path="suicide_dataset.csv"):
    """
        ### Load the dataset from a local file or download it if not available.

        >>> Args:
            dataset_path (str, optional): Path to the dataset file. Defaults to "suicide_dataset.csv".

        >>> Returns:
            pandas.DataFrame: Loaded and preprocessed dataset.
    """
    try:
        if os.path.exists(dataset_path):
            print("File exists")
        else:
            dataset = load_dataset("Ram07/Detection-for-Suicide")
            df = pd.DataFrame(dataset['train'])
            df.to_csv(dataset_path, index=False)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit()

    finally:
        try:
            data = pd.read_csv(dataset_path)
            print(f"Database {dataset_path} loaded ...._")
            data = clean_na(data)
            return clean_dataset(data)
        except Exception as e:
            print("Error loading dataset:", e)
            sys.exit()

def preprocess_text(texts):
    """
        ### Preprocess the text data by tokenizing, converting to lowercase, and removing stop words and non-alphanumeric characters.
        >>> >>> preprocess_text("took rest sleeping pills painkillers i want to end struggle of past 6 years")
        >>> >>> "took rest sleeping pills painkillers want end struggle past 6 years"  # <----- this is in the str format

        Args:
            texts (list or np.ndarray): Input texts to preprocess.

        Returns:
            np.ndarray: Preprocessed texts.
    """
    # stop_words = set(stopwords.words('english'))
    # tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
    # filtered_tokens = [[word for word in tokens if word.isalnum() and word not in stop_words] for tokens in tokenized_texts]
    # preprocessed_texts = [' '.join(tokens) for tokens in filtered_tokens]
    # return np.array(preprocessed_texts)
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(texts.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)


def train_model_SDM(X_train, y_train):
    """
        ### Train the SVM model on the given training data.

        >>> Args:
            X_train (pandas.Series or np.ndarray): Input features for training.
            y_train (pandas.Series or np.ndarray): Target labels for training.

        >>> Returns:
            sklearn.pipeline.Pipeline: Trained SVM model pipeline.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    ngram_range = (1, 3)  # Using trigrams
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.9, min_df=5)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = SVC(kernel='linear', C=10, verbose=True, probability=True)
    model.fit(X_train_vec, y_train)
    print("Model trained successfully.")
    
    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    print(report)

    save_model(model)
    return model

def save_model(model):
    """
        ### Save the trained model to a file.

        >>> Args:
            model (sklearn.pipeline.Pipeline): Trained model pipeline.
    """
    try:
        joblib.dump(model, 'suicide_detection_model.pkl')
        print("Model saved successfully.")
    except Exception as e:
        print("Error saving model:", e)

def clean_na(data):
    """
        ### Remove rows with NaN values from the dataset.

        >>> Args:
            data (pandas.DataFrame): Input dataset.

        >>> Returns:
            pandas.DataFrame: Dataset with NaN values removed.
    """
    cleaned_data = data.dropna()
    return cleaned_data

def clean_dataset(dataset):
    """
        ### Convert the target labels to binary values (0 for "suicide", 1 for "non-suicide").

        >>> Args:
            dataset (pandas.DataFrame): Input dataset.

        >>> Returns:
            pandas.DataFrame: Dataset with binary target labels.
    """
    binary_convert = lambda x: 0 if x == "suicide" else 1
    dataset['class'] = dataset['class'].apply(binary_convert)
    return dataset

def start_model(X_train, y_train, train_model=False):
    """
        ### Load or train the SVM model based on the provided parameters.

        >>> Args:
            X_train (pandas.Series or np.ndarray): Input features for training.
            y_train (pandas.Series or np.ndarray): Target labels for training.
            train_model (bool, optional): Flag to train a new model or load an existing one. Defaults to False.

        >>> Returns:
            sklearn.pipeline.Pipeline: Trained or loaded SVM model.
            TfidfVectorizer: Fitted vectorizer used for training.
    """
    print('start_model'.center(120, "-"))
    if not train_model and os.path.exists("suicide_detection_model.pkl"):
        try:
            model = joblib.load('suicide_detection_model.pkl')
            vectorizer = joblib.load('suicide_detection_vectorizer.pkl')  # Load the vectorizer
            return model, vectorizer
        except Exception as e:
            print("Error loading the model:", e)
    else:
        print("starting model training.........")
        model = train_model_SDM(X_train, y_train)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train)
        joblib.dump(model, 'suicide_detection_model.pkl')  # Save the trained model
        joblib.dump(vectorizer, 'suicide_detection_vectorizer.pkl')  # Save the fitted vectorizer
        return model, vectorizer

def preprocess_text_with_progress(data, text_column='text', label_column='class', save_file=None):
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        if pd.isna(text):  # Check for NaN values
            return ''  # Replace NaN values with an empty string
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)

    def update_progress_bar(loading):
        loading.show('Preprocessing Text...', finish_message='Preprocessing Finished!✅')

    # Start a thread for the progress bar
    loading = TermLoading()
    pbar_thread = threading.Thread(target=update_progress_bar, args=(loading,))
    pbar_thread.start()

    processed_text = data[text_column].apply(preprocess_text)
    
    # Combine preprocessed text with labels
    processed_data = pd.concat([processed_text, data[label_column]], axis=1)

    if save_file:
        processed_data.to_csv(save_file, index=False)

    # Update the progress bar to indicate completion
    loading.finished = True

    # Wait for the progress bar thread to finish
    pbar_thread.join()

    return processed_data

def evaluate_model_score(model, X_test, y_test):
    try:
        accuracy = model.score(X_test, y_test)
        print("\nModel Accuracy on Test Set:", round(accuracy * 100, 3), "%")
    except Exception as e:
        print("Error evaluating model:", e)
        sys.exit()

def evaluating_SDM(model, X_test, y_test):
    # Remove samples with NaN labels
    X_test = X_test.dropna()
    y_test = y_test.dropna()

def print_matrix_img(y_test, prediction):
    # Assuming 'y_test' and 'prediction' are available from the main function
    def create_confusion_matrix(y_test, prediction):
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        confusion_matrix_data = {
            'True Positive': tp,
            'True Negative': tn,
            'False Positive': fp,
            'False Negative': fn
        }
        return confusion_matrix_data

    # Call this function with 'y_test' and 'prediction' from the main function
    conf_matrix_data = create_confusion_matrix(y_test, prediction)
    print(conf_matrix_data)


def create_confusion_matrix_from_csv(csv_path):
    data = load_dataset_download("suicide_dataset.csv")
    # Preprocess text
    data_processed = preprocess_text_with_progress(data, text_column='text', label_column='class')

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_processed['text'], data_processed['class'], test_size=0.2, random_state=42)

    model = start_model(X_train, y_train,train_model=False)

    # Make predictions
    print("generating PREDICTIONS ____________")
    prediction = model.predict(X_test)
    
    # Convert predicted labels to integers
    prediction_int = [0 if label == 'non-suicide' else 1 for label in prediction]

    # Create and display confusion matrix
    print_matrix_img(y_test, prediction_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, prediction_int), annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    print("calling evalute function")
    evaluate_model_performance(model,X_test,y_test,prediction_int)
    # result
    # {'True Positive': 1111, 'True Negative': 1297, 'False Positive': 12098, 'False Negative': 20382}

def evaluate_model_performance(model, X_test, y_test,prediction_int):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    # Make predictions
    y_pred = prediction_int
    # Compute precision
    precision = precision_score(y_test, y_pred)
    # Compute recall
    recall = recall_score(y_test, y_pred)
    # Compute F1-score
    f1 = f1_score(y_test, y_pred)
    # Compute ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Predict probabilities for positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    def update_progress_bar(term_loading):
        term_loading.show('Evaluating Model...', finish_message='Evaluation Finished!✅')
        evaluate_model_score(model, X_test, y_test)
        term_loading.finished = True

    # Start a thread for the progress bar
    term_loading = TermLoading()
    progress_bar_thread = Thread(target=update_progress_bar, args=(term_loading,))
    progress_bar_thread.start()

    # Wait for the progress bar thread to finish
    progress_bar_thread.join()


    # Print precision, recall, and F1-score
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()



def main():
    load_necessary_lib()
    data = load_dataset_download("suicide_dataset.csv")
    print("------------>>>>> Data loaded <<<<<-----------")

    # Preprocess text
    data_processed = preprocess_text_with_progress(data, text_column='text', label_column='class', save_file='final_cleaned_processed_text.csv')

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_processed['text'], data_processed['class'], test_size=0.2, random_state=42)
    
    print("Data split done.")
    start_time = time.time()
    # Train or load the model
    # model = start_model(X_train, y_train,train_model=False)
    model, vectorizer = start_model(X_train, y_train, train_model=False)
#-------------------------------------------------------------------
    # vectorizer = TfidfVectorizer()
    # # Fit the vectorizer on the training data
    # vectorizer.fit(X_train)
    # joblib.dump(vectorizer, 'suicide_detection_vectorizer.pkl')
#-------------------------------------------------------------------
    end_time = time.time()
    print(f"Total model trining time: {end_time - start_time} seconds")
    # Evaluate the model
    # evaluating_SDM(model, X_test, y_test)

    cont = True
    while cont:
        # Accept user input
        # try:
        user_input = input("Enter your response: ").strip()
        if user_input == "quit":
            cont = False
            del model, data,X_test,X_train,y_test,y_train,data_processed
        elif user_input:
            preprocessed_input = preprocess_text(user_input)
            input_features = vectorizer.transform([preprocessed_input])
            prediction = model.predict(input_features)[0]
            prediction_scores = model.predict_proba(input_features)[0]
            print("Prediction Score for Suicidal:  {:.2f}%".format(prediction_scores[1] * 100).rjust(100, " "))
            print("Prediction Score for Non-Suicidal:  {:.2f}%".format(prediction_scores[0] * 100).rjust(100, " "))
            print(f"Prediction: >> {prediction}\n".rjust(120, " "))
        else:
            print("Empty input. Please provide a response.\n".center(100,"="))
        # except Exception as e:
        #     print("Error processing user input:", e)


# # if __name__ == "__main__":
# try:
#     main()
#     # create_confusion_matrix_from_csv("suicide_dataset.csv")
# except KeyboardInterrupt:
#     print("\nExiting the program.")
# except Exception as e:
#     print("An error occurred:", e)
#     sys.exit()  # Uncomment this line if you want to exit on error




main()