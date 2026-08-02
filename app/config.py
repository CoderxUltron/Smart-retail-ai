"""Central configuration, loaded from environment variables / .env file."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    api_key: str = "demo-secret-key"
    model_dir: str = str(ROOT_DIR / "app" / "models")
    data_dir: str = str(ROOT_DIR / "data")
    img_size: int = 96
    face_match_threshold: float = 70.0
    chatbot_confidence_threshold: float = 0.35

    class Config:
        env_file = ".env"


settings = Settings()
