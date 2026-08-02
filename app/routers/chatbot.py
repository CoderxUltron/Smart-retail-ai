"""Endpoint for Module B3 — /chatbot."""
from fastapi import APIRouter, Depends

from app.schemas import ChatIn, ChatOut
from app.security import require_api_key
from app.services import chatbot_service

router = APIRouter(tags=["Chatbot"], dependencies=[Depends(require_api_key)])


@router.post("/chatbot", response_model=ChatOut)
async def chatbot(payload: ChatIn):
    return chatbot_service.chatbot_reply(payload.message)
