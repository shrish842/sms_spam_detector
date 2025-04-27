# SMS Spam Detection Project

This project implements a machine learning model to classify SMS messages as either "spam" or "ham" (not spam). It utilizes Natural Language Processing (NLP) techniques for text preprocessing and classic classification algorithms for prediction.

## Table of Contents

*   [Description](#description)
*   [Features](#features)
*   [Dataset](#dataset)
*   [Installation](#installation)
*   [Usage](#usage)
*   [File Structure](#file-structure)
*   [Model Evaluation](#model-evaluation)
*   [Future Improvements](#future-improvements)
*   [License](#license)

## Description

The goal of this project is to build a robust classifier capable of distinguishing spam SMS messages from legitimate ones (ham). The process involves:

1.  Loading and cleaning the SMS data.
2.  Preprocessing the text messages (lowercase, remove punctuation/numbers, remove stopwords, lemmatization).
3.  Converting text data into numerical features using TF-IDF vectorization.
4.  Training multiple classification models (Multinomial Naive Bayes, Logistic Regression, Linear SVM).
5.  Evaluating model performance using metrics like Accuracy, Precision, Recall, and F1-Score, with a focus on detecting spam effectively.
6.  Providing a function to classify new, unseen SMS messages.

**Technologies Used:**
*   Python 3.x
*   Pandas
*   NLTK (Natural Language Toolkit)
*   Scikit-learn
*   Matplotlib & Seaborn (for visualization)

## Features

*   Loads data from the standard SMS Spam Collection Dataset.
*   Performs thorough text preprocessing tailored for SMS messages.
*   Uses TF-IDF for effective text feature extraction.
*   Trains and evaluates three common text classification models.
*   Provides detailed evaluation metrics, including classification reports and confusion matrices.
*   Includes a function to predict whether new SMS messages are spam or ham.
*   Visualizes the class distribution and confusion matrices.

## Dataset

This project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

*   **Source:** [Kaggle Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
*   **Description:** The dataset contains 5,572 SMS messages in English, tagged according to being ham (legitimate) or spam.
*   **Columns:**
    *   `label`: 'ham' or 'spam'
    *   `message`: The raw text content of the SMS message.

The dataset is included in this repository as `spam.csv`.

## Installation

To run this project locally, follow these steps:

1.  **Prerequisites:**
    *   Python 3.7+ installed
    *   Git installed

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```
    *(Replace `YOUR_USERNAME/YOUR_REPOSITORY_NAME` with your actual repo details, e.g., `shrish842/sms_spam_detector`)*

3.  **Create and activate a virtual environment (recommended):**
    *   **Windows:**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

4.  **Install required dependencies:**
    ```bash
    pip install pandas numpy nltk scikit-learn matplotlib seaborn
    ```

5.  **Download NLTK data:**
    Run Python in your activated environment and download the necessary NLTK resources:
    ```bash
    python
    ```
    Then inside the Python interpreter (`>>>` prompt):
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    exit()
    ```

## Usage

To run the SMS spam detection script:

1.  Make sure your virtual environment is activated.
2.  Navigate to the project directory in your terminal.
3.  Execute the main script:
    ```bash
    python spam_detector.py
    ```

**Expected Output:**

*   Information about data loading and preprocessing printed to the console.
*   Class distribution plot displayed in a separate window.
*   Model training progress messages.
*   Evaluation metrics (Accuracy, Precision, Recall, F1-Score) and Classification Reports printed for each model.
*   Confusion Matrix plots displayed for each model (you may need to close one plot window to see the next).
*   A summary table comparing model performance.
*   Predictions ('Spam' or 'Ham') for several example new messages printed at the end.

**Predicting New Messages:**
The script includes a `predict_spam(sms_message)` function. You can modify the script to input your own messages or integrate this function into another application.

## File Structure


## Model Evaluation

The script trains and evaluates the following models:

*   Multinomial Naive Bayes
*   Logistic Regression
*   Linear Support Vector Machine (SVM)

Evaluation focuses on Precision, Recall, and F1-Score for the 'Spam' class, as correctly identifying spam while minimizing misclassification of legitimate messages is crucial. Detailed results and confusion matrices are printed upon running `spam_detector.py`. The best-performing model based on the F1 score is highlighted.

## Future Improvements

*   **Hyperparameter Tuning:** Use techniques like GridSearchCV or RandomizedSearchCV to find optimal parameters for the models and TF-IDF vectorizer.
*   **Different Vectorizers:** Experiment with CountVectorizer or word embeddings (like Word2Vec, GloVe, or Sentence-BERT).
*   **More Advanced Models:** Explore deep learning models like LSTMs, GRUs, or Transformers (BERT) for potentially higher accuracy.
*   **Cross-Validation:** Implement k-fold cross-validation for more robust evaluation.
*   **Deployment:** Wrap the prediction logic in a simple web application using Flask or Streamlit.
*   **Error Analysis:** Investigate the messages that the best model misclassifies to understand its weaknesses.

## License

This project is likely under the MIT License (or choose another if you prefer). You can create a `LICENSE` file with the chosen license text. For now:

Distributed under the MIT License. See `LICENSE` file for more information (if available).
