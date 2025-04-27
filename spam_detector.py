# -----------------------------------------------------------------------------
# Step 1: Import Necessary Libraries
# -----------------------------------------------------------------------------
# Import pandas for data manipulation and reading CSV files.
# We use DataFrames to hold and organize our data easily.
import pandas as pd

# Import numpy for numerical operations (though less used directly here, pandas relies on it).
import numpy as np

# Import the regular expression module 're' for text cleaning (removing punctuation, numbers).
import re

# Import Natural Language Toolkit 'nltk' for text processing tasks.
import nltk

# Import stopwords (common words like 'the', 'is', 'in') from nltk.corpus to remove them later.
# These words usually don't help distinguish spam from ham.
from nltk.corpus import stopwords

# Import WordNetLemmatizer to reduce words to their base or dictionary form (e.g., 'running' -> 'run').
# Lemmatization is generally preferred over stemming as it produces actual words.
from nltk.stem import WordNetLemmatizer

# --- Download NLTK data ---
# These lines download data required by NLTK's functions.
# 'punkt' is for tokenization (splitting text into words/sentences).
# 'stopwords' is the list of common English words to ignore.
# 'wordnet' is the lexical database needed for lemmatization.
# NOTE: You only need to run these ONCE per Python environment.
# After the first successful run, you should COMMENT THEM OUT (add '#' at the start)
# or remove them to avoid downloading every time.

# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#      nltk.download('wordnet')
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#      nltk.download('stopwords')
# try:
#      nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#      nltk.download('punkt')



# Import train_test_split to divide our data into training and testing sets.
# This is crucial to evaluate how well our model generalizes to unseen data.
from sklearn.model_selection import train_test_split

# Import TfidfVectorizer to convert text data into numerical feature vectors using TF-IDF.
# TF-IDF (Term Frequency-Inverse Document Frequency) is effective for text classification
# because it highlights words that are important to a specific document within a larger collection.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import classification models from scikit-learn.
# MultinomialNB (Naive Bayes) is a probabilistic classifier often used as a strong baseline for text data.
from sklearn.naive_bayes import MultinomialNB
# LogisticRegression is a robust linear model for binary classification tasks.
from sklearn.linear_model import LogisticRegression
# LinearSVC (Support Vector Classifier with a linear kernel) is another powerful and efficient linear model.
from sklearn.svm import LinearSVC

# Import metrics to evaluate model performance.
# accuracy_score: Overall percentage of correct predictions.
# classification_report: Provides precision, recall, F1-score, and support for each class.
# confusion_matrix: Shows True Positives, True Negatives, False Positives, False Negatives.
# precision_score: Measures the accuracy of positive predictions (how many predicted spam were actually spam).
# recall_score: Measures how many actual positives were correctly identified (how many actual spam messages were found).
# f1_score: Harmonic mean of precision and recall, good balanced metric for imbalanced datasets.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Import libraries for plotting graphs.
# matplotlib.pyplot is the standard plotting library.
import matplotlib.pyplot as plt
# seaborn builds on matplotlib and provides more attractive statistical visualizations.
import seaborn as sns

print("Libraries imported successfully!")

# -----------------------------------------------------------------------------
# Step 2: Load and Explore Data
# -----------------------------------------------------------------------------
print("\n--- Loading Data ---")
# Define the path to the dataset file. Since this script is in the SAME folder
# as 'spam.csv', we just need the filename.
file_path = 'spam.csv'

# Use pandas read_csv function to load the data into a DataFrame.
# Specify encoding='latin-1' because this specific dataset often contains characters
# not handled by the default 'utf-8' encoding, causing errors if not specified.
try:
    df = pd.read_csv(file_path, encoding='latin-1')
    print(f"Data loaded successfully from {file_path}")
except FileNotFoundError:
    # Handle the error if the file isn't found where expected.
    print(f"Error: File not found at {file_path}. Make sure 'spam.csv' is in the same folder as the script.")
    exit() # Stop the script if data can't be loaded.
except Exception as e:
    # Catch other potential errors during file reading.
    print(f"An error occurred while reading the file: {e}")
    exit()

print("\n--- Initial Data Exploration ---")
# Display the first 5 rows of the DataFrame to get a quick look at the data structure and content.
print("First 5 rows of data:")
print(df.head())

# Print concise summary information about the DataFrame, including column names, data types, and non-null counts.
# Useful for identifying missing values and understanding column types.
print("\nDataFrame Info:")
df.info()

# --- Initial Data Cleaning ---
# This dataset sometimes includes extra, unnamed columns from the CSV export process.
# We create a list of columns containing 'Unnamed' in their name.
cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
# If such columns exist...
if cols_to_drop:
    # ...drop them from the DataFrame. 'inplace=False' (default) returns a new DataFrame.
    df = df.drop(columns=cols_to_drop)
    print(f"\nDropped unnecessary columns: {cols_to_drop}")

# Rename the default columns 'v1' and 'v2' to more descriptive names 'label' and 'message'.
# This makes the code easier to read and understand.
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
print("\n--- Columns Renamed ---")
print("First 5 rows after renaming:")
print(df.head())

# Map the categorical labels 'ham' and 'spam' to numerical values 0 and 1 respectively.
# Machine learning models require numerical input for the target variable.
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print("\n--- Labels Mapped to Numerical ---")
# Display the original label and the new numerical label for the first few rows.
print(df[['label', 'label_num']].head())

# Check for any missing values in each column. isnull() returns True/False, sum() counts True values.
print("\n--- Missing Values Check ---")
print(df.isnull().sum())
# This dataset typically has no missing values in the essential columns.

# --- Check Class Distribution ---
# See how many messages belong to each class ('ham' vs 'spam').
print("\n--- Class Distribution ---")
print(df['label'].value_counts())
# Calculate and print the percentage of messages that are spam.
# This helps understand if the dataset is imbalanced (it usually is, with more ham).
print("\nSpam percentage:", round(df['label_num'].mean() * 100, 2), "%")

# Visualize the class distribution using a count plot from seaborn.
# This gives a clear visual representation of the imbalance.
print("\n--- Visualizing Class Distribution ---")
plt.figure(figsize=(6, 4)) # Set the figure size for better readability.
sns.countplot(x='label', data=df, palette=['skyblue', 'salmon']) # Create the plot, assign colors.
plt.title('Class Distribution (Ham vs Spam)') # Set the title of the plot.
plt.xlabel('Message Type') # Label the x-axis.
plt.ylabel('Count') # Label the y-axis.
plt.show() # Display the plot (might open in a separate window).

# -----------------------------------------------------------------------------
# Step 3: Text Preprocessing
# -----------------------------------------------------------------------------
print("\n--- Setting up Text Preprocessing ---")
# Initialize the WordNetLemmatizer. We'll use its .lemmatize() method.
lemmatizer = WordNetLemmatizer()
# Get the set of standard English stopwords from NLTK. Using a set provides faster lookups.
stop_words = set(stopwords.words('english'))

# Define a function to apply all preprocessing steps to a single text message.
def preprocess_text(text):
    # 1. Convert the text to lowercase. This ensures 'Free' and 'free' are treated as the same word.
    text = text.lower()
    # 2. Remove punctuation, numbers, and any characters that are not lowercase letters or whitespace.
    # The regular expression '[^a-z\s]' matches anything that is NOT a lowercase letter (a-z) or whitespace (\s).
    # re.sub replaces these matched characters with an empty string ''.
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenize the cleaned text into a list of individual words.
    # nltk.word_tokenize handles various cases like contractions intelligently (though less relevant after removing punctuation).
    words = nltk.word_tokenize(text)
    # 4. Remove stopwords and 5. Lemmatize the remaining words.
    # Use a list comprehension for efficiency:
    # - Iterate through each 'word' in the 'words' list.
    # - Include the word only if it's 'not in stop_words'.
    # - Also include only if 'len(word) > 1' to remove single letters potentially left after cleaning.
    # - Apply 'lemmatizer.lemmatize(word)' to each included word.
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    # Join the processed words back into a single string, separated by spaces.
    # Vectorizers like TF-IDF expect string input.
    return ' '.join(processed_words)

# Apply the preprocessing function to every message in the 'message' column.
# The result is stored in a new column called 'cleaned_message'.
print("Applying preprocessing to messages...")
df['cleaned_message'] = df['message'].apply(preprocess_text)
print("Preprocessing complete.")

# Display examples to show the effect of preprocessing.
print("\nSample messages before and after cleaning:")
print("Original [0]:", df['message'][0])
print("Cleaned  [0]:", df['cleaned_message'][0])
print("\nOriginal [2]:", df['message'][2]) # Example spam message
print("Cleaned  [2]:", df['cleaned_message'][2])

# -----------------------------------------------------------------------------
# Step 4: Feature Extraction (TF-IDF)
# -----------------------------------------------------------------------------
print("\n--- Feature Extraction (TF-IDF) ---")
# Initialize the TfidfVectorizer.
# max_features=3000 limits the vocabulary size to the 3000 most frequent words across all messages.
# This helps reduce dimensionality and computational cost, and can prevent overfitting on rare words.
# This value (3000) is a hyperparameter that can be tuned.
tfidf_vectorizer = TfidfVectorizer(max_features=3000)

# Define the feature set (X) as the cleaned text messages.
X = df['cleaned_message']
# Define the target variable (y) as the numerical labels (0 for ham, 1 for spam).
y = df['label_num']

# -----------------------------------------------------------------------------
# Step 5: Split Data into Training and Testing Sets
# -----------------------------------------------------------------------------
print("\n--- Splitting Data ---")
# Use train_test_split to divide the data.
# X: features (cleaned messages)
# y: target labels (0 or 1)
# test_size=0.20: Allocate 20% of the data for the test set, 80% for training.
# random_state=42: Ensures that the split is the same every time the code is run (reproducibility).
# stratify=y: Crucial for imbalanced datasets. Ensures that the proportion of 'ham' and 'spam'
#             in the training set and test set is the same as in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} messages")
print(f"Test set size: {X_test.shape[0]} messages")

# --- Apply TF-IDF Vectorization ---
print("\n--- Applying TF-IDF to Data ---")
# Fit the TF-IDF vectorizer *only* on the training data (X_train).
# This learns the vocabulary (the mapping of words to feature indices) and calculates
# the Inverse Document Frequency (IDF) weights based *only* on the training text.
# Then, transform X_train into a numerical TF-IDF matrix.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data (X_test) using the *already fitted* vectorizer.
# This uses the vocabulary and IDF learned from the training data to convert
# the test messages into a numerical matrix with the same feature columns.
# IMPORTANT: Do NOT fit again on the test data to prevent data leakage.
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Print the shape of the resulting TF-IDF matrices (rows=messages, columns=features/words).
print(f"Shape of TF-IDF matrix (Train): {X_train_tfidf.shape}")
print(f"Shape of TF-IDF matrix (Test): {X_test_tfidf.shape}")

# -----------------------------------------------------------------------------
# Step 6: Model Selection and Training
# -----------------------------------------------------------------------------
print("\n--- Model Selection and Training ---")
# Initialize the chosen classification models with default or specified parameters.
# MultinomialNB is parameter-free in its basic form for this application.
mnb = MultinomialNB()
# LogisticRegression: solver='liblinear' is a good choice for binary classification and smaller datasets.
# random_state ensures reproducibility if there's any randomness in the solver.
log_reg = LogisticRegression(solver='liblinear', random_state=42)
# LinearSVC: random_state for reproducibility. dual="auto" selects optimization algorithm based on data size/features.
svm = LinearSVC(random_state=42, dual="auto")

# Create a dictionary to hold the models for easy iteration during training and evaluation.
models = {
    "Multinomial Naive Bayes": mnb,
    "Logistic Regression": log_reg,
    "Support Vector Machine (Linear)": svm # Changed name slightly for clarity
}

# Train each model using the training data.
# The .fit() method trains the model by learning the relationship between
# the TF-IDF features (X_train_tfidf) and the target labels (y_train).
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# -----------------------------------------------------------------------------
# Step 7: Model Evaluation
# -----------------------------------------------------------------------------
print("\n--- Evaluating Models on Test Set ---")

# Create an empty dictionary to store the evaluation results for each model.
results = {}

# Iterate through each trained model to evaluate its performance on the unseen test set.
for name, model in models.items():
    print(f"\n--- Results for: {name} ---")
    # Use the trained model's .predict() method to get predictions for the test features (X_test_tfidf).
    y_pred = model.predict(X_test_tfidf)

    # Calculate various performance metrics by comparing the true labels (y_test) with the predictions (y_pred).
    accuracy = accuracy_score(y_test, y_pred)
    # Use pos_label=1 for precision, recall, f1 to focus on the 'spam' class (label 1).
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Store the calculated metrics in the results dictionary.
    results[name] = {'Accuracy': accuracy, 'Precision (Spam)': precision, 'Recall (Spam)': recall, 'F1 Score (Spam)': f1}

    # Print the calculated metrics, formatted to 4 decimal places.
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Spam): {precision:.4f}")
    print(f"Recall (Spam): {recall:.4f}")
    print(f"F1 Score (Spam): {f1:.4f}")

    # Print the detailed classification report, which includes metrics for both 'ham' (0) and 'spam' (1).
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

    # Calculate the confusion matrix.
    cm = confusion_matrix(y_test, y_pred)
    # Visualize the confusion matrix using a heatmap for better interpretation.
    plt.figure(figsize=(6, 4)) # Set figure size.
    # annot=True displays the counts in each cell. fmt='d' formats numbers as integers. cmap='Blues' sets the color scheme.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Label') # Label axes.
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}') # Set title specific to the model.
    plt.show() # Display the plot.

# --- Compare Model Results ---
print("\n--- Overall Model Comparison ---")
# Convert the results dictionary into a pandas DataFrame for easy comparison.
# .T transposes the DataFrame so model names are rows and metrics are columns.
results_df = pd.DataFrame(results).T
print(results_df)

# Identify the best model based on the F1 Score for the Spam class.
# F1-score is often a good metric for imbalanced classification like spam detection.
# .idxmax() finds the index (model name in this case) corresponding to the maximum value in the F1 Score column.
best_model_name = results_df['F1 Score (Spam)'].idxmax()
print(f"\nBest performing model based on F1 Score (Spam): {best_model_name}")

# -----------------------------------------------------------------------------
# Step 8: Making Predictions on New Messages
# -----------------------------------------------------------------------------
print("\n--- Making Predictions on New Data ---")

# Select the best model identified in the previous step from our 'models' dictionary.
best_model = models[best_model_name]
print(f"Using '{best_model_name}' for new predictions.")

# Define a function to take a raw SMS message string and predict if it's spam or ham.
def predict_spam(sms_message):
    # 1. Preprocess the input message using the SAME preprocessing function used for training data.
    cleaned_message = preprocess_text(sms_message)
    # 2. Transform the cleaned message into a TF-IDF vector using the SAME vectorizer fitted on the training data.
    # Note: vectorizer.transform expects a list or iterable, so we put the single message in square brackets [].
    message_tfidf = tfidf_vectorizer.transform([cleaned_message])
    # 3. Use the selected best model to predict the class (0 or 1) for the transformed message vector.
    # .predict() returns an array, so we take the first element [0].
    prediction_num = best_model.predict(message_tfidf)[0]
    # 4. Convert the numerical prediction (0 or 1) back into a human-readable label ('Ham' or 'Spam').
    return 'Spam' if prediction_num == 1 else 'Ham'

# Define a list of example new messages to test the prediction function.
new_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    "Hey, are you coming to the party tonight?",
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
    "Sorry, I'll be late for the meeting.",
    "Congratulations! You've been selected for a free cruise to the Bahamas. Reply YES to claim.",
    "Can you pick up some milk on your way home?",
    "Your account security code is 883721. Do not share this code."
]

# Loop through the example messages and print the prediction for each one.
for msg in new_messages:
    prediction = predict_spam(msg)
    print(f"Message: '{msg}'\nPrediction: >>> {prediction} <<<\n")

print("--- Script Finished ---")