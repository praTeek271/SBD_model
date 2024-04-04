# DESIGNING A MODEL FOR SUICIDAL BEHAVIOUR DETECTION 

## Introduction
This project aims to develop a machine learning model for the detection of suicidal tendencies in text data. The model utilizes natural language processing techniques and support vector machines (SVM) for classification.

## Dataset
The dataset used for training and testing the model consists of text samples labeled as either "suicidal" or "non-suicidal". The dataset is preprocessed to remove noise and irrelevant information.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- nltk
- joblib
- tqdm

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/prateek271/SBD_model.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python suicide_detection_main.py
   ```

4. Follow the instructions to input text data and receive predictions.

## Model Evaluation
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and graphs are generated to visualize the results.

## Results and Discussion
The model demonstrates promising performance in detecting suicidal tendencies in text data. However, further optimization and fine-tuning may be required to improve its accuracy and response rate.

## Future Work
- Incorporate more advanced machine learning algorithms such as CNN and BiLSTM.
- Explore ensemble learning techniques for improved model performance.
- Gather additional labeled data to enhance model training.

## Contributors
- Prateek Kumar Singh
- Ritika Singh
- Shadiya Khan
- Bharkha Bharatwaj

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
