import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# ──────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────
print("Loading dataset...")
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fake_df = pd.read_csv(os.path.join(BASE_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(BASE_DIR, "True.csv"))
fake_df["label"] = 1   # 1 = Fake
true_df["label"] = 0   # 0 = Real

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[["title", "text", "label"]]
df["content"] = df["title"] + " " + df["text"]
df = df.dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"Total samples: {len(df)}")
print(f"Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")

# ──────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ──────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # remove punctuation/numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

print("Preprocessing text (this may take a minute)...")
df["cleaned"] = df["content"].apply(clean_text)

# ──────────────────────────────────────────────
# 3. FEATURE EXTRACTION (TF-IDF)
# ──────────────────────────────────────────────
print("Extracting features with TF-IDF...")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["cleaned"])
y = df["label"]

# ──────────────────────────────────────────────
# 4. TRAIN MODEL
# ──────────────────────────────────────────────
print("Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 5. EVALUATE
# ──────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ──────────────────────────────────────────────
# 6. SAVE MODEL + VECTORIZER
# ──────────────────────────────────────────────
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("\n✅ Model saved as 'fake_news_model.pkl'")
print("✅ Vectorizer saved as 'tfidf_vectorizer.pkl'")