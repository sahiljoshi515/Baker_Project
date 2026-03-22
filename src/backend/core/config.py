"""
This file is set up to provide the settings from our .env file (useful during testing, 
as it's very easy to override with your own custom get_setting impl.)
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Baker_Project root Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]



class Settings(BaseSettings):

    api_title: str = "Digital Collections Explorer API"
    api_description: str = "API for OCR, metadata extraction, and document persistence"
    api_version: str = "0.1.0"

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    app_env: str = Field(default="development", validation_alias="APP_ENV")

    database_url: str = Field(default="sqlite:///database.db", validation_alias="DATABASE_URL")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        validation_alias="CORS_ORIGINS",
    )

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    mistral_api_key: str | None = Field(default=None, validation_alias="MISTRAL_API_KEY")
    openai_chat_model: str = Field(default="gpt-4", validation_alias="OPENAI_CHAT_MODEL")
    mistral_ocr_model: str = Field(default="mistral-ocr-latest", validation_alias="MISTRAL_OCR_MODEL")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: str | None = Field(default="app.log", validation_alias="LOG_FILE")
    frontend_dist_dir: str | None = Field(default=PROJECT_ROOT / "frontend.py", validation_alias="FRONTEND_DIST_DIR")

    # clip_model: str = "openai/clip-vit-base-patch32"
    # device: str = "cuda"
    # batch_size: int = 32
    # collection_type: str = "photographs"
    # raw_data_dir: str = "data/raw"
    # processed_data_dir: str = "data/processed"
    # embeddings_dir: str = "data/embeddings"
    # thumbnails_dir: str = "data/thumbnails"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return json.loads(stripped)
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        raise TypeError("CORS_ORIGINS must be a list or comma-separated string")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
