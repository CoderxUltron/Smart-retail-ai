"""Module B — NLP service: text preprocessing + sentiment analysis.

Loads artifacts produced by notebooks/03_sentiment_model_training.ipynb.
"""
import re
import string
from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from app.config import settings

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

_MODEL_DIR = Path(settings.model_dir)
_sentiment_model = None
_vectorizer = None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [_LEMMATIZER.lemmatize(t) for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)


def _load_sentiment_model():
    global _sentiment_model, _vectorizer
    if _sentiment_model is None:
        model_path = _MODEL_DIR / "sentiment_model.pkl"
        vec_path = _MODEL_DIR / "vectorizer.pkl"
        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError(
                f"{model_path} / {vec_path} not found — run notebooks/03_sentiment_model_training.ipynb first."
            )
        _sentiment_model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)
    return _sentiment_model, _vectorizer


def analyze_sentiment(text: str) -> dict:
    model, vectorizer = _load_sentiment_model()
    vec = vectorizer.transform([clean_text(text)])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return {"sentiment": pred, "confidence": round(float(proba), 3)}
