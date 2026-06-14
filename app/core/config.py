"""
Core configuration management using Pydantic Settings.
All settings are loaded from environment variables with .env file support.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- App ---
    app_name: str = Field(default="ConversationalAI", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="logs", env="LOG_DIR")
    log_file: str = Field(default="app.log", env="LOG_FILE")
    log_backup_count: int = Field(default=7, env="LOG_BACKUP_COUNT")

    # --- Server ---
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # --- CORS ---
    allowed_origins: list[str] = Field(
        default=["*"], env="ALLOWED_ORIGINS"
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", env="OPENAI_BASE_URL"
    )
    openai_chat_model: str = Field(default="gpt-4o-mini", env="OPENAI_CHAT_MODEL")
    openai_timeout: float = Field(default=30.0, env="OPENAI_TIMEOUT")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")

    # --- STT (Speech-to-Text) ---
    stt_provider: str = Field(default="openai_whisper", env="STT_PROVIDER")
    stt_model: str = Field(default="whisper-1", env="STT_MODEL")

    # --- TTS (Text-to-Speech) ---
    tts_provider: str = Field(default="openai_tts", env="TTS_PROVIDER")
    tts_model: str = Field(default="tts-1", env="TTS_MODEL")
    tts_voice: str = Field(default="alloy", env="TTS_VOICE")
    tts_audio_format: str = Field(default="mp3", env="TTS_AUDIO_FORMAT")

    # --- Session Memory ---
    session_max_messages: int = Field(default=20, env="SESSION_MAX_MESSAGES")
    session_ttl_seconds: int = Field(default=3600, env="SESSION_TTL_SECONDS")

    # --- Request ---
    request_timeout: float = Field(default=60.0, env="REQUEST_TIMEOUT")
    max_audio_size_mb: int = Field(default=25, env="MAX_AUDIO_SIZE_MB")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance — call this via FastAPI Depends."""
    return Settings()
