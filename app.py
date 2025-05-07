from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load pre-trained model and vectorizer
clf = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Extract keywords for contextual insights
def extract_keywords(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    return ", ".join(keywords[:5])  # Return top 5 keywords

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the user
    review = request.form['review']
    
    # Vectorize the input text
    X_input = vectorizer.transform([review])
    
    # Predict sentiment (0: negative, 1: neutral, 2: positive)
    sentiment = clf.predict(X_input)[0]

    # Get keywords from the review
    keywords = extract_keywords(review)

    sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return render_template('index.html', sentiment=sentiment_label[sentiment], keywords=keywords, review=review)

if __name__ == '__main__':
    app.run(debug=True)
