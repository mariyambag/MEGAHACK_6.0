Fake News Detector - ML-Powered Web Application

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen?style=flat-square)

A web application that uses **Machine Learning** and **Natural Language Processing (NLP)** to detect whether a news article is **Real or Fake** - with a confidence score.

---

Features

- **Paste Text** analyze any news headline or article directly
- **URL Checker** paste a news URL and it auto-scrapes and analyzes the article
- **Confidence Score** shows fake/real probability with animated progress bars
- **99% Accuracy** trained on 44,000+ labeled news articles
- **Clean UI** dark-themed, responsive frontend

---

#Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | Logistic Regression (scikit-learn) |
| Feature Extraction | TF-IDF Vectorizer |
| NLP Preprocessing | NLTK (stopwords, lemmatization) |
| Backend API | Flask + Flask-CORS |
| Web Scraping | BeautifulSoup4 + Requests |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

#Project Structure

```
FAKE_NEWS_DETECTION/
└── MODEL/
    ├── app.py                  ← Flask REST API (backend)
    ├── trainmodel.py           ← Model training script
    ├── fake_news_model.pkl     ← Trained ML model (generated)
    ├── tfidf_vectorizer.pkl    ← TF-IDF vectorizer (generated)
    ├── Fake.csv                ← Fake news dataset
    ├── True.csv                ← Real news dataset
    └── index.html              ← Frontend UI
```

---

#Dataset

**Source:** [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| Property | Value |
|----------|-------|
| Total Samples | ~44,000 articles |
| Fake News | ~23,000 articles |
| Real News | ~21,000 articles |
| Source | Reuters, PolitiFact |

---

#How It Works

```
User Input (Text / URL)
        ↓
Text Cleaning (lowercase, remove URLs, punctuation, stopwords)
        ↓
Lemmatization (running → run, governments → government)
        ↓
TF-IDF Vectorization (text → numbers)
        ↓
Logistic Regression Model
        ↓
Prediction: FAKE or REAL + Confidence Score
        ↓
Displayed on Frontend
```

---

#Model Performance

| Metric | Real | Fake |
|--------|------|------|
| Precision | 0.98 | 0.99 |
| Recall | 0.99 | 0.99 |
| F1-Score | 0.99 | 0.99 |
| **Overall Accuracy** | **99%** | |

---

#Installation & Setup

### Step 1 - Clone or download the project

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### Step 2 — Install dependencies

```bash
pip install flask flask-cors scikit-learn pandas nltk joblib requests beautifulsoup4
```

### Step 3 — Download the dataset

Download `Fake.csv` and `True.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them in the `MODEL/` folder.

### Step 4 — Train the model

```bash
cd MODEL
python trainmodel.py
```

This generates `fake_news_model.pkl` and `tfidf_vectorizer.pkl`.

### Step 5 — Start the Flask backend

```bash
python app.py
```

Flask runs on: `http://localhost:5000`

### Step 6 — Open the frontend

Double-click `index.html` in your browser. Done! 🎉

---

## API Endpoints

### `GET /`
Health check
```json
{ "status": "Fake News Detector API is running!" }
```

### `POST /predict`
Predict from text
```json
// Request
{ "text": "Government secretly controls all media!" }

// Response
{
  "prediction": "Fake",
  "confidence": 94.2,
  "fake_probability": 94.2,
  "real_probability": 5.8
}
```

### `POST /predict-url`
Predict from a news URL
```json
// Request
{ "url": "https://example.com/news-article" }

// Response
{
  "prediction": "Real",
  "confidence": 97.1,
  "fake_probability": 2.9,
  "real_probability": 97.1,
  "scraped_text_preview": "Article text preview..."
}
```

---

## Future Improvements

- [ ] BERT / Transformer-based model for better accuracy
- [ ] Multi-language support
- [ ] Source credibility scoring
- [ ] Browser extension
- [ ] Explainable AI — highlight suspicious words

---

## Author

Mariyam,Sanskruti,Shruti,Dnyanada
College Project — Fake News Detection using ML & NLP

---

