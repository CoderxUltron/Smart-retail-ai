"""Module A1 — reusable OpenCV preprocessing utilities."""
import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img: np.ndarray, width: int = 300) -> np.ndarray:
    h, w = img.shape[:2]
    ratio = width / w
    return cv2.resize(img, (width, int(h * ratio)))


def apply_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def canny_edges(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = to_grayscale(img) if img.ndim == 3 else img
    return cv2.Canny(gray, low, high)


def detect_faces(img: np.ndarray):
    """Returns (annotated_image, list_of_face_boxes) using a Haar cascade."""
    gray = to_grayscale(img) if img.ndim == 3 else img
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    out = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, faces


def crop_largest_face(img: np.ndarray):
    """Detects the largest face in the image and returns its grayscale crop, or None."""
    gray = to_grayscale(img) if img.ndim == 3 else img
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y:y + h, x:x + w]
