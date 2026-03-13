from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
model = joblib.load(os.path.join(BASE_DIR, "fake_news_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
print("Model loaded!")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def scrape_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        if len(text.strip()) < 100:
            return None, "Could not extract enough text from this URL."
        return text, None
    except Exception as e:
        return None, f"Error fetching URL: {str(e)}"

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    label = "Fake" if prediction == 1 else "Real"
    confidence = round(float(max(probabilities)) * 100, 2)
    fake_prob = round(float(probabilities[1]) * 100, 2)
    real_prob = round(float(probabilities[0]) * 100, 2)
    return {
        "prediction": label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob
    }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide text."}), 400
    text = data["text"].strip()
    if len(text) < 20:
        return jsonify({"error": "Text is too short."}), 400
    return jsonify(predict_news(text))

@app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Please provide a URL."}), 400
    text, error = scrape_url(data["url"].strip())
    if error:
        return jsonify({"error": error}), 400
    result = predict_news(text)
    result["scraped_text_preview"] = text[:300] + "..."
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Fake News Detector API is running!"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
