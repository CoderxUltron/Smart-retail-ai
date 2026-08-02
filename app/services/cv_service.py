"""Module A — Computer Vision service: product classification + face recognition.

Loads trained artifacts produced by notebooks/01_image_classifier_training.ipynb
and notebooks/02_face_recognition_setup.ipynb. Run those notebooks first so the
files referenced below exist under app/models/.
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services.cv_utils import crop_largest_face

_MODEL_DIR = Path(settings.model_dir)

_product_classifier = None
_class_names = None
_face_recognizer = None
_face_db_meta = None
_visit_log: list[dict] = []


def _load_product_classifier():
    global _product_classifier, _class_names
    if _product_classifier is None:
        import tensorflow as tf  # imported lazily so the API can boot without TF for NLP-only use
        model_path = _MODEL_DIR / "product_classifier.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found — run notebooks/01_image_classifier_training.ipynb first."
            )
        _product_classifier = tf.keras.models.load_model(model_path)
        _class_names = json.loads((_MODEL_DIR / "class_names.json").read_text())
    return _product_classifier, _class_names


def _load_face_recognizer():
    global _face_recognizer, _face_db_meta
    if _face_recognizer is None:
        model_path = _MODEL_DIR / "face_recognizer.yml"
        db_path = _MODEL_DIR / "face_db.pkl"
        if not model_path.exists() or not db_path.exists():
            raise FileNotFoundError(
                f"{model_path} / {db_path} not found — run notebooks/02_face_recognition_setup.ipynb first."
            )
        _face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        _face_recognizer.read(str(model_path))
        _face_db_meta = pickle.loads(db_path.read_bytes())
    return _face_recognizer, _face_db_meta


def classify_product(image_bgr: np.ndarray) -> dict:
    model, class_names = _load_product_classifier()
    import tensorflow as tf
    img = cv2.resize(image_bgr, (settings.img_size, settings.img_size))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype("float32"))
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx = int(np.argmax(pred))
    return {"category": class_names[idx], "confidence": float(pred[idx])}


def recognize_face(image_bgr: np.ndarray) -> dict:
    recognizer, db_meta = _load_face_recognizer()
    face_crop = crop_largest_face(image_bgr)
    if face_crop is None:
        result = {"customer": "unknown", "status": "no_face_detected", "confidence_distance": -1.0}
        _visit_log.append({**result, "timestamp": str(datetime.now())})
        return result

    label, confidence = recognizer.predict(face_crop)
    is_known = confidence < settings.face_match_threshold
    result = {
        "customer": db_meta["names"][label] if is_known else "unknown",
        "status": "returning_customer" if is_known else "new_visitor",
        "confidence_distance": round(float(confidence), 2),
    }
    _visit_log.append({**result, "timestamp": str(datetime.now())})
    return result


def dashboard_stats() -> dict:
    return {
        "total_visits_logged": len(_visit_log),
        "returning_customers": sum(1 for v in _visit_log if v["status"] == "returning_customer"),
        "new_visitors": sum(1 for v in _visit_log if v["status"] == "new_visitor"),
    }
