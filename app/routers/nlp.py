"""Endpoint for Module B2 — /analyze-sentiment."""
from fastapi import APIRouter, Depends

from app.schemas import SentimentOut, TextIn
from app.security import require_api_key
from app.services import nlp_service

router = APIRouter(tags=["NLP"], dependencies=[Depends(require_api_key)])


@router.post("/analyze-sentiment", response_model=SentimentOut)
async def analyze_sentiment(payload: TextIn):
    return nlp_service.analyze_sentiment(payload.text)
