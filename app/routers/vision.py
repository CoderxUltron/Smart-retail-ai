"""Endpoints for Module A — /classify-product and /recognize-face."""
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.schemas import FaceOut, ProductOut
from app.security import require_api_key
from app.services import cv_service

router = APIRouter(tags=["Computer Vision"], dependencies=[Depends(require_api_key)])


def _decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return img


@router.post("/classify-product", response_model=ProductOut)
async def classify_product(file: UploadFile = File(...)):
    img = _decode_image(await file.read())
    return cv_service.classify_product(img)


@router.post("/recognize-face", response_model=FaceOut)
async def recognize_face(file: UploadFile = File(...)):
    img = _decode_image(await file.read())
    return cv_service.recognize_face(img)
