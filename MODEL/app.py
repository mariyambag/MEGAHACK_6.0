from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, re, os, nltk, requests, feedparser
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

# ── PASTE YOUR NEWSAPI KEY HERE ──
NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"
# ────────────────────────────────

CREDIBLE_SOURCES = [
    'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com',
    'theguardian.com', 'nytimes.com', 'washingtonpost.com',
    'ndtv.com', 'thehindu.com', 'hindustantimes.com',
    'timesofindia.com', 'aljazeera.com', 'bloomberg.com',
    'cnbc.com', 'cnn.com', 'npr.org'
]

# ── HELPERS ─────────────────────────────────────────────

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def predict_news(text):
    cleaned    = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probs      = model.predict_proba(vectorized)[0]
    return {
        "prediction":       "Fake" if prediction == 1 else "Real",
        "confidence":       round(float(max(probs)) * 100, 2),
        "fake_probability": round(float(probs[1]) * 100, 2),
        "real_probability": round(float(probs[0]) * 100, 2)
    }

def scrape_url(url):
    try:
        r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all('p'))
        if len(text.strip()) < 100:
            return None, "Could not extract enough text."
        return text, None
    except Exception as e:
        return None, str(e)

def check_real_time(headline):
    try:
        query = headline[:150].replace(' ', '+')
        feed  = feedparser.parse(
            f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        )
        hits = []
        for entry in feed.entries[:10]:
            src = entry.get('source', {}).get('href', '') or entry.get('link', '')
            for domain in CREDIBLE_SOURCES:
                if domain in src:
                    hits.append({
                        'title':     entry.get('title', ''),
                        'source':    src,
                        'published': entry.get('published', 'N/A')
                    })
                    break
        verdict = ('LIKELY REAL' if len(hits) >= 2 else
                   'UNVERIFIED'  if len(hits) == 1  else
                   'NOT FOUND IN NEWS')
        return {
            'sources_found':     hits[:3],
            'credible_hits':     len(hits),
            'real_time_verdict': verdict
        }
    except:
        return {'sources_found': [], 'credible_hits': 0, 'real_time_verdict': 'SEARCH UNAVAILABLE'}

def combined_verdict(ml_prediction, rt_verdict):
    """
    Combine ML result + Real-Time source check into one final verdict.

    ML=Real  + Sources Found     → REAL        (double confirmed)
    ML=Fake  + Not Found         → FAKE        (double confirmed)
    ML=Real  + Not Found         → UNVERIFIED  (ML says real but no sources)
    ML=Fake  + Sources Found     → UNVERIFIED  (ML says fake but sources exist)
    Anything + Search Unavailable → trust ML only
    """
    if rt_verdict == 'SEARCH UNAVAILABLE':
        return ml_prediction.upper(), "Search unavailable — ML verdict only."

    if ml_prediction == 'Real' and rt_verdict == 'LIKELY REAL':
        return 'REAL', "ML model and credible news sources both confirm this is genuine."

    if ml_prediction == 'Fake' and rt_verdict == 'NOT FOUND IN NEWS':
        return 'FAKE', "ML model flagged this and no credible sources found — likely misinformation."

    if ml_prediction == 'Real' and rt_verdict == 'NOT FOUND IN NEWS':
        return 'UNVERIFIED', "ML model says real but no credible sources found. Could be new or niche news."

    if ml_prediction == 'Fake' and rt_verdict == 'LIKELY REAL':
        return 'UNVERIFIED', "ML model flagged this but credible sources are reporting it. Manual review recommended."

    if rt_verdict == 'UNVERIFIED':
        return 'UNVERIFIED', "Limited source coverage found. Treat with caution."

    return ml_prediction.upper(), "Based on available data."

# ── ROUTES ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FakeXpose API running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text'."}), 400
    text = data["text"].strip()
    if len(text) < 20:
        return jsonify({"error": "Text too short."}), 400

    ml     = predict_news(text)
    rt     = check_real_time(text)
    final, reason = combined_verdict(ml["prediction"], rt["real_time_verdict"])

    return jsonify({
        # ML results
        "ml_prediction":    ml["prediction"],
        "confidence":       ml["confidence"],
        "fake_probability": ml["fake_probability"],
        "real_probability": ml["real_probability"],
        # Real-time results
        "real_time_verdict": rt["real_time_verdict"],
        "credible_sources":  rt["sources_found"],
        "credible_hits":     rt["credible_hits"],
        # Combined final verdict
        "final_verdict":     final,
        "verdict_reason":    reason
    })

@app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Please provide 'url'."}), 400
    text, error = scrape_url(data["url"].strip())
    if error:
        return jsonify({"error": error}), 400

    ml     = predict_news(text)
    rt     = check_real_time(text[:200])
    final, reason = combined_verdict(ml["prediction"], rt["real_time_verdict"])

    return jsonify({
        "ml_prediction":         ml["prediction"],
        "confidence":            ml["confidence"],
        "fake_probability":      ml["fake_probability"],
        "real_probability":      ml["real_probability"],
        "real_time_verdict":     rt["real_time_verdict"],
        "credible_sources":      rt["sources_found"],
        "credible_hits":         rt["credible_hits"],
        "final_verdict":         final,
        "verdict_reason":        reason,
        "scraped_text_preview":  text[:300] + "..."
    })

def fetch_articles(params):
    resp = requests.get(
        "https://newsapi.org/v2/top-headlines",
        params={**params, "apiKey": NEWS_API_KEY},
        timeout=10
    )
    data = resp.json()
    if data.get("status") != "ok":
        return []
    return data.get("articles", [])

@app.route("/latest-news", methods=["GET"])
def latest_news():
    if NEWS_API_KEY == "YOUR_NEWSAPI_KEY_HERE":
        return jsonify({"error": "Add your NewsAPI key in app.py (line 20)."}), 400
    try:
        india_articles  = fetch_articles({"country": "in", "pageSize": 10})
        global_articles = fetch_articles({
            "language": "en", "pageSize": 10,
            "sources": "bbc-news,reuters,al-jazeera-english,associated-press,cnn,bloomberg"
        })
        seen, combined = set(), []
        for article in india_articles + global_articles:
            headline = (article.get("title") or "").strip()
            if not headline or headline == "[Removed]" or headline in seen:
                continue
            seen.add(headline)
            desc  = article.get("description") or ""
            ml    = predict_news(headline + " " + desc)
            # For latest news, use ML only (no RT search to avoid rate limiting)
            combined.append({
                "headline":         headline,
                "source":           article.get("source", {}).get("name", "Unknown"),
                "url":              article.get("url", ""),
                "published":        article.get("publishedAt", ""),
                "prediction":       ml["prediction"],
                "confidence":       ml["confidence"],
                "fake_probability": ml["fake_probability"],
                "real_probability": ml["real_probability"]
            })
        return jsonify({"articles": combined, "total": len(combined)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
