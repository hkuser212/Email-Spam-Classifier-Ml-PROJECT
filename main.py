import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
# Try reading the CSV with different encodings
  # Change this to the absolute path of your file

try:
    data = pd.read_csv('spam.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('spam.csv', encoding='latin1')

# Keep only the necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Encode the labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Identify non-string entries in 'text' column
non_string_entries = data[~data['text'].apply(lambda x: isinstance(x, str))]
print("Non-string entries in 'text' column before conversion:")
print(non_string_entries)

# Convert all entries in the 'text' column to strings
data['text'] = data['text'].astype(str)

# Re-check for non-string entries after conversion
non_string_entries = data[~data['text'].apply(lambda x: isinstance(x, str))]
print("Non-string entries in 'text' column after conversion:")
print(non_string_entries)

# Download stopwords
nltk.download('stopwords')

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = text.strip()
    return text

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

# Fit and transform the text data
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: \n{conf_matrix}")
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')
