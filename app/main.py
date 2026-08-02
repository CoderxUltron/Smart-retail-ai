"""FastAPI entrypoint — Smart Retail & Customer Intelligence Platform.

Run locally:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive Swagger docs.
"""
from fastapi import FastAPI

from app.routers import chatbot, dashboard, nlp, vision

app = FastAPI(
    title="Smart Retail & Customer Intelligence API",
    description="Face recognition, product classification, sentiment analysis, and FAQ chatbot behind one API.",
    version="1.0.0",
)

app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)
app.include_router(dashboard.router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "Smart Retail AI API is running", "docs": "/docs"}
