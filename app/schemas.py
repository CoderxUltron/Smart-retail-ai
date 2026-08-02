"""Pydantic request/response models for the API."""
from pydantic import BaseModel, Field


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["The delivery was fast and the quality is great!"])


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, examples=["Where is my order?"])


class ProductOut(BaseModel):
    category: str
    confidence: float


class FaceOut(BaseModel):
    customer: str
    status: str  # "returning_customer" | "new_visitor"
    confidence_distance: float


class SentimentOut(BaseModel):
    sentiment: str  # "positive" | "negative"
    confidence: float


class ChatOut(BaseModel):
    intent: str
    confidence: float
    reply: str


class DashboardStatsOut(BaseModel):
    total_visits_logged: int
    returning_customers: int
    new_visitors: int
