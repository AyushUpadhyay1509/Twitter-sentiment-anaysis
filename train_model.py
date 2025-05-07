import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the model

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Preprocessing Tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & Stopword removal
    return text

# Load dataset
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) == 2:
                texts.append(clean_text(parts[0]))  # Clean text
                labels.append(parts[1].strip())  # Extract label
    return texts, labels

# Load train dataset
train_texts, train_labels = load_data("train.txt")

# Convert to DataFrame
df = pd.DataFrame({'text': train_texts, 'label': train_labels})

# Encode Labels
df['label'] = df['label'].astype('category').cat.codes  # Convert emotion labels to numbers

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# Convert Text to TF-IDF Vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_tfidf, y_train)

# Predict on Test Data
y_pred = model.predict(X_test_tfidf)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save Model & Vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
