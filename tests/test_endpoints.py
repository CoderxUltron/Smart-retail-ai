"""API tests. These mock the ML services so they run without needing trained
model files on disk — train the real models via the notebooks/ before a
manual/integration test pass.
"""
import io

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services import chatbot_service, cv_service, nlp_service

client = TestClient(app)
HEADERS = {"x-api-key": settings.api_key}


def _fake_jpeg_bytes():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(buf.tobytes())


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"].startswith("Smart Retail")


def test_missing_api_key_rejected():
    r = client.post("/analyze-sentiment", json={"text": "great!"})
    assert r.status_code == 401


def test_analyze_sentiment(monkeypatch):
    monkeypatch.setattr(
        nlp_service, "analyze_sentiment",
        lambda text: {"sentiment": "positive", "confidence": 0.91},
    )
    r = client.post("/analyze-sentiment", json={"text": "Loved it!"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["sentiment"] == "positive"


def test_chatbot(monkeypatch):
    monkeypatch.setattr(
        chatbot_service, "chatbot_reply",
        lambda msg: {"intent": "order_status", "confidence": 0.8, "reply": "Track it in My Orders."},
    )
    r = client.post("/chatbot", json={"message": "where is my order"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["intent"] == "order_status"


def test_classify_product(monkeypatch):
    monkeypatch.setattr(
        cv_service, "classify_product",
        lambda img: {"category": "electronics", "confidence": 0.77},
    )
    r = client.post(
        "/classify-product",
        files={"file": ("sample.jpg", _fake_jpeg_bytes(), "image/jpeg")},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["category"] == "electronics"


def test_recognize_face(monkeypatch):
    monkeypatch.setattr(
        cv_service, "recognize_face",
        lambda img: {"customer": "unknown", "status": "new_visitor", "confidence_distance": 55.2},
    )
    r = client.post(
        "/recognize-face",
        files={"file": ("sample.jpg", _fake_jpeg_bytes(), "image/jpeg")},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "new_visitor"


def test_dashboard_stats(monkeypatch):
    monkeypatch.setattr(
        cv_service, "dashboard_stats",
        lambda: {"total_visits_logged": 3, "returning_customers": 1, "new_visitors": 2},
    )
    r = client.get("/dashboard/stats", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["total_visits_logged"] == 3
