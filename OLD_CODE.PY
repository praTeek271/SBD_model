#--------------------------------------------------------------------------------------------------
    # import pandas as pd
    # import nltk
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.svm import SVC
    # from sklearn.model_selection import train_test_split
    # from nltk.corpus import stopwords
    # from sklearn.pipeline import make_pipeline
    # import joblib
    # import os
    # import sys
    # from tqdm import tqdm
    # import time
    # from datasets import load_dataset



    # def load_necessary_lib():
    #     # Preload NLTK data
    #     nltk.download('stopwords')
    #     nltk.download('punkt')
    #     print("Libraries loaded successfully")

    # def load_dataset_download(dataset_path="suicide_dataset.csv"):
    #     """
    #     loading dataset from the web if not present else , loading from the local directory
    #     """
    #     # Load the dataset if not present
    #     try:
    #         if os.path.exists(dataset_path):
    #             print("File exists")
    #         else:
    #             dataset = load_dataset("Ram07/Detection-for-Suicide")
    #             df = pd.DataFrame(dataset['train'])
    #             df.to_csv(dataset_path, index=False)
    #     except Exception as e:
    #         print("Error loading dataset:", e)
    #         sys.exit()

    #     finally:
    #         # Load data directly from the CSV file
    #         try:
    #             data = pd.read_csv(dataset_path)
    #             print(f"Database {dataset_path} loaded ...._")
    #             data = clean_na(data)
    #             return clean_dataset(data)
    #         except Exception as e:
    #             print("Error loading dataset:", e)
    #             sys.exit()

    # def preprocess_text(text):
    #     """
    #     # Preprocess the text data :
    #     >>> >>> preprocess_text("took rest sleeping pills painkillers i want to end struggle of past 6 years")
    #     >>> >>> "took rest sleeping pills painkillers want end struggle past 6 years"  # <----- this is in the str format
    #     """
    #     stop_words = set(stopwords.words('english'))
    #     tokens = nltk.word_tokenize(text.lower())  # Tokenization and convert to lowercase
    #     filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    #     return ' '.join(filtered_tokens)

    # def train_model_SDM(X_train, y_train):
    #     vectorizer = TfidfVectorizer()
    #     classifier = SVC(kernel='linear', verbose=True, probability=True)  # Enable probability estimates
    #     model = make_pipeline(vectorizer, classifier)
    #     model.fit(X_train, y_train)
    #     print("Model trained successfully.")
    #     save_model(model)
    #     return model

    # def save_model(model):     
    #     # Save the trained model
    #     try:
    #         joblib.dump(model, 'suicide_detection_model.pkl')
    #         print("Model saved successfully.")
    #     except Exception as e:
    #         print("Error saving model:", e)

    # def clean_na(data):
    #     # Remove rows with NaN values
    #     cleaned_data = data.dropna()
    #     return cleaned_data

    # def clean_dataset(dataset):
    #     binary_convert = lambda x: 0 if x == "suicide" else 1
    #     # data=pd.read_csv("suicide_dataset.csv")
    #     dataset['class'] = dataset['class'].apply(binary_convert)
    #     return dataset


    # def start_model(X_train, y_train, train_model=False):
    #     print('start_model'.center(120, "-"))
    #     if not train_model and os.path.exists("suicide_detection_model.pkl"):
    #         # Load the saved model
    #         try:
    #             model = joblib.load('suicide_detection_model.pkl')
    #             return model
    #         except Exception as e:
    #             print("Error loading the model:", e)
    #     else:
    #         model = train_model_SDM(X_train, y_train)
    #         return model

    # def preprocess_text_with_progress(data, text_column='text', label_column='class', save_file=None):
    #     stop_words = set(stopwords.words('english'))

    #     def preprocess_text(text):
    #         if pd.isna(text):  # Check for NaN values
    #             return ''  # Replace NaN values with an empty string
    #         tokens = nltk.word_tokenize(text.lower())
    #         filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    #         return ' '.join(filtered_tokens)

    #     tqdm.pandas(desc="Preprocessing Text")
    #     processed_text = data[text_column].progress_apply(preprocess_text)
        
    #     # Combine preprocessed text with labels
    #     processed_data = pd.concat([processed_text, data[label_column]], axis=1)

    #     if save_file:
    #         processed_data.to_csv(save_file, index=False)

    #     return processed_data

    # def evaluating_SDM(model, X_test, y_test):
    #     # Remove samples with NaN labels
    #     X_test = X_test[~y_test.isna()]
    #     y_test = y_test.dropna()

    #     # Evaluate the model on the test set with a spinning animation
    #     with tqdm(total=len(X_test), desc="Evaluating Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
    #         for _ in range(len(X_test)):
    #             time.sleep(0.1)  # Simulate evaluation time (remove in actual usage)
    #             pbar.update(1)
        
    #     try:
    #         accuracy = model.score(X_test, y_test)
    #         print("\nModel Accuracy on Test Set:", round(accuracy * 100, 3), "%")
    #     except Exception as e:
    #         print("Error evaluating model:", e)

    # def main():
    #     load_necessary_lib()
    #     data = load_dataset_download("suicide_dataset.csv")
    #     print("------------>>>>> Data loaded <<<<<-----------")

    #     # Preprocess text
    #     data_processed = preprocess_text_with_progress(data, text_column='text', label_column='class', save_file='final_cleaned_processed_text.csv')

    #     # Split data into train and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(data_processed['text'], data_processed['class'], test_size=0.2, random_state=42)
    #     print("Data split done.")

    #     # Train or load the model
    #     model = start_model(X_train, y_train, train_model=False)

    #     # Evaluate the model
    #     evaluating_SDM(model, X_test, y_test)

    #     cont = True
    #     while cont:
    #         # Accept user input
    #         try:
    #             user_input = input("Enter your response: ").strip()
    #             if user_input == "quit":
    #                 cont = False
    #             elif user_input:
    #                 # Preprocess user input
    #                 preprocessed_input = preprocess_text(user_input)

    #                 # Predict
    #                 prediction = model.predict([preprocessed_input])[0]
                    
    #                 # Output prediction result
    #                 prediction_scores = model.predict_proba([preprocessed_input])[0]
    #                 print("Prediction Score for Suicidal:", "{:.2f}%".format(prediction_scores[1] * 100))
    #                 print("Prediction Score for Non-Suicidal:", "{:.2f}%".format(prediction_scores[0] * 100))
    #                 # Output prediction result
    #                 predicted_label = 'suicidal' if prediction == 1 else 'non-suicidal'
    #                 print("Prediction:", predicted_label)
    #             else:
    #                 print("Empty input. Please provide a response.")
    #         except Exception as e:
    #             print("Error processing user input:", e)

    # if __name__ == "__main__":
    #     try:
    #         main()
    #     except KeyboardInterrupt:
    #         print("\nExiting the program.")
    #     except Exception as e:
    #         print("An error occurred:", e)
    #         sys.exit()  # Uncomment this line if you want to exit on error
#--------------------------------------------------------------------------------------------------
# import threading
# from threading import Thread
# import pandas as pd
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
# from sklearn.pipeline import make_pipeline
# import joblib
# import os
# import sys
# from TermLoading import TermLoading  # Import the TermLoading class
# import time
# from datasets import load_dataset
# # from keras.models import load_model

# # Exception handling decorator
# def log_exceptions(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             exc_type, exc_obj, exc_tb = sys.exc_info()
#             print(f"Exception '{e}' occurred in function '{func.__name__}' at line {exc_tb.tb_lineno}")
#     return wrapper

# def load_necessary_lib():
#     # Preload NLTK data
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     print("Libraries loaded successfully")

# def load_dataset_download(dataset_path="suicide_dataset.csv"):
#     try:
#         if os.path.exists(dataset_path):
#             print("File exists")
#         else:
#             dataset = load_dataset("Ram07/Detection-for-Suicide")
#             df = pd.DataFrame(dataset['train'])
#             df.to_csv(dataset_path, index=False)
#     except Exception as e:
#         print("Error loading dataset:", e)
#         sys.exit()

#     finally:
#         try:
#             data = pd.read_csv(dataset_path)
#             print(f"Database {dataset_path} loaded ...._")
#             data = clean_na(data)
#             return clean_dataset(data)
#         except Exception as e:
#             print("Error loading dataset:", e)
#             sys.exit()

# def preprocess_text(text):
#     """
#         # Preprocess the text data :
#         >>> >>> preprocess_text("took rest sleeping pills painkillers i want to end struggle of past 6 years")
#         >>> >>> "took rest sleeping pills painkillers want end struggle past 6 years"  # <----- this is in the str format
#     """
#     stop_words = set(stopwords.words('english'))
#     tokens = nltk.word_tokenize(text.lower())
#     filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#     return ' '.join(filtered_tokens)

# def train_model_SDM(X_train, y_train):
#     vectorizer = TfidfVectorizer()
#     classifier = SVC(kernel='linear', verbose=True, probability=True)
#     model = make_pipeline(vectorizer, classifier)
#     model.fit(X_train, y_train)
#     print("Model trained successfully.")
#     save_model(model)
#     return model

# def save_model(model):     
#     try:
#         model.save('suicide_detection_model.h5')
#         joblib.dump(model, 'suicide_detection_model.pkl')
#         print("Model saved successfully.")
#     except Exception as e:
#         print("Error saving model:", e)

# def clean_na(data):
#     cleaned_data = data.dropna()
#     return cleaned_data

# def clean_dataset(dataset):
#     binary_convert = lambda x: 0 if x == "suicide" else 1
#     dataset['class'] = dataset['class'].apply(binary_convert)
#     return dataset

# def start_model(X_train, y_train, train_model=False):
#     print('start_model'.center(120, "-"))
#     if not train_model and os.path.exists("suicide_detection_model.pkl"):
#         try:
#             model = joblib.load('suicide_detection_model.pkl')
#             # model = load_model('suicide_detection_model.h5')
#             return model
#         except Exception as e:
#             print("Error loading the model:", e)
#     else:
#         model = train_model_SDM(X_train, y_train)
#         return model

# def preprocess_text_with_progress(data, text_column='text', label_column='class', save_file=None):
#     stop_words = set(stopwords.words('english'))

#     def preprocess_text(text):
#         if pd.isna(text):  # Check for NaN values
#             return ''  # Replace NaN values with an empty string
#         tokens = nltk.word_tokenize(text.lower())
#         filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#         return ' '.join(filtered_tokens)

#     def update_progress_bar(loading):
#         loading.show('Preprocessing Text...', finish_message='Preprocessing Finished!✅')

#     # Start a thread for the progress bar
#     loading = TermLoading()
#     pbar_thread = threading.Thread(target=update_progress_bar, args=(loading,))
#     pbar_thread.start()

#     processed_text = data[text_column].apply(preprocess_text)
    
#     # Combine preprocessed text with labels
#     processed_data = pd.concat([processed_text, data[label_column]], axis=1)

#     if save_file:
#         processed_data.to_csv(save_file, index=False)

#     # Update the progress bar to indicate completion
#     loading.finished = True

#     # Wait for the progress bar thread to finish
#     pbar_thread.join()

#     return processed_data

# def evaluate_model_score(model, X_test, y_test):
#     try:
#         accuracy = model.score(X_test, y_test)
#         print("\nModel Accuracy on Test Set:", round(accuracy * 100, 3), "%")
#     except Exception as e:
#         print("Error evaluating model:", e)
#         sys.exit()

# def evaluating_SDM(model, X_test, y_test):
#     # Remove samples with NaN labels
#     X_test = X_test.dropna()
#     y_test = y_test.dropna()

#     def update_progress_bar(term_loading):
#         term_loading.show('Evaluating Model...', finish_message='Evaluation Finished!✅')
#         evaluate_model_score(model, X_test, y_test)
#         term_loading.finished = True

#     # Start a thread for the progress bar
#     term_loading = TermLoading()
#     progress_bar_thread = Thread(target=update_progress_bar, args=(term_loading,))
#     progress_bar_thread.start()

#     # Wait for the progress bar thread to finish
#     progress_bar_thread.join()

# def main():
#     load_necessary_lib()
#     data = load_dataset_download("suicide_dataset.csv")
#     print("------------>>>>> Data loaded <<<<<-----------")

#     # Preprocess text
#     data_processed = preprocess_text_with_progress(data, text_column='text', label_column='class', save_file='final_cleaned_processed_text.csv')

#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data_processed['text'], data_processed['class'], test_size=0.2, random_state=42)
#     print("Data split done.")

#     # Train or load the model
#     model = start_model(X_train, y_train, train_model=False)

#     # Evaluate the model
#     # evaluating_SDM(model, X_test, y_test)

#     cont = True
#     while cont:
#         # Accept user input
#         try:
#             user_input = input("Enter your response: ").strip()
#             if user_input == "quit":
#                 cont = False
#             elif user_input:
#                 # Preprocess user input
#                 preprocessed_input = preprocess_text(user_input)

#                 # Predict
#                 prediction = model.predict([preprocessed_input])[0]
                
#                 # Output prediction result
#                 prediction_scores = model.predict_proba([preprocessed_input])[0]
#                 print("Prediction Score for Suicidal:  {:.2f}%".format(prediction_scores[1] * 100).rjust(100," "))
#                 print("Prediction Score for Non-Suicidal:  {:.2f}%".format(prediction_scores[0] * 100).rjust(100," "))
#                 # Output prediction result
#                 # predicted_label = 'suicidal' if prediction == 0 else 'non-suicide'
#                 print(f"Prediction: >> {prediction}\n".rjust(120," "))
#             else:
#                 print("Empty input. Please provide a response.\n".center(100,"="))
#         except Exception as e:
#             print("Error processing user input:", e)


# def print_matrix_img(y_test, prediction):
#     from sklearn.metrics import confusion_matrix

#     # Assuming 'y_test' and 'prediction' are available from the main function
#     def create_confusion_matrix(y_test, prediction):
#         tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
#         confusion_matrix_data = {
#             'True Positive': tp,
#             'True Negative': tn,
#             'False Positive': fp,
#             'False Negative': fn
#         }
#         return confusion_matrix_data

#     # Call this function with 'y_test' and 'prediction' from the main function
#     conf_matrix_data = create_confusion_matrix(y_test, prediction)
#     print(conf_matrix_data)



# # if __name__ == "__main__":
# try:
#     main()
#     # data,X_train, X_test, y_train, y_test,model,prediction,prediction_scores=main()
#     # print_matrix_img(y_test, prediction)
# except KeyboardInterrupt:
#     print("\nExiting the program.")
# except Exception as e:
#     print("An error occurred:", e)
#     sys.exit()  # Uncomment this line if you want to exit on error


