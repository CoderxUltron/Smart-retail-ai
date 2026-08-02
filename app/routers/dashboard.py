"""Endpoint for Module C — /dashboard/stats."""
from fastapi import APIRouter, Depends

from app.schemas import DashboardStatsOut
from app.security import require_api_key
from app.services import cv_service

router = APIRouter(tags=["Dashboard"], dependencies=[Depends(require_api_key)])


@router.get("/dashboard/stats", response_model=DashboardStatsOut)
async def dashboard_stats():
    return cv_service.dashboard_stats()
