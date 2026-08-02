"""Module B3 — Hybrid rule-based + ML FAQ chatbot service.

Loads artifacts produced by notebooks/03_sentiment_model_training.ipynb
(the same notebook trains the chatbot intent classifier).
"""
import json
import random
from pathlib import Path

import joblib

from app.config import settings
from app.services.nlp_service import clean_text

_MODEL_DIR = Path(settings.model_dir)
_DATA_DIR = Path(settings.data_dir)

_intent_classifier = None
_intent_vectorizer = None
_intents = None

_RULE_KEYWORDS = {
    "order_status": ["track", "order", "shipment", "package"],
    "return_policy": ["return", "refund", "exchange"],
    "store_hours": ["hours", "open", "close", "timing"],
    "payment_methods": ["payment", "pay", "card", "upi", "cod"],
}


def _load():
    global _intent_classifier, _intent_vectorizer, _intents
    if _intent_classifier is None:
        model_path = _MODEL_DIR / "chatbot_model.pkl"
        vec_path = _MODEL_DIR / "chatbot_vectorizer.pkl"
        intents_path = _DATA_DIR / "intents.json"
        if not model_path.exists() or not vec_path.exists() or not intents_path.exists():
            raise FileNotFoundError(
                f"Chatbot artifacts not found — run notebooks/03_sentiment_model_training.ipynb first."
            )
        _intent_classifier = joblib.load(model_path)
        _intent_vectorizer = joblib.load(vec_path)
        _intents = json.loads(intents_path.read_text())
    return _intent_classifier, _intent_vectorizer, _intents


def _rule_based_fallback(text: str):
    text = text.lower()
    for tag, keywords in _RULE_KEYWORDS.items():
        if any(k in text for k in keywords):
            return tag
    return None


def chatbot_reply(message: str) -> dict:
    classifier, vectorizer, intents = _load()
    vec = vectorizer.transform([clean_text(message)])
    proba = classifier.predict_proba(vec)[0]
    idx = int(proba.argmax())
    tag, confidence = classifier.classes_[idx], float(proba[idx])

    if confidence < settings.chatbot_confidence_threshold:
        fallback_tag = _rule_based_fallback(message)
        tag = fallback_tag if fallback_tag else "unknown"

    if tag == "unknown":
        reply = "Sorry, I didn't quite get that. Could you rephrase, or ask about orders, returns, hours, or payments?"
    else:
        reply = random.choice(intents[tag]["responses"])

    return {"intent": tag, "confidence": round(confidence, 3), "reply": reply}
