import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
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

# Convert all entries in the 'text' column to strings
data['text'] = data['text'].astype(str)

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_app.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load('spam_classifier_app.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.title("Email Spam Classifier")

st.write("""
This is a simple email spam classifier app. Enter an email text below to classify it as spam or ham.
""")

email_text = st.text_area("Enter email text:")

if st.button("Classify"):
    if email_text:
        cleaned_text = preprocess_text(email_text)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]
        result = "Spam" if prediction == 1 else "Ham"
        st.write(f"The email is classified as: {result}")
    else:
        st.write("Please enter some text to classify.")
