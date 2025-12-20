"""アプリケーション設定"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定"""

    # Application
    APP_NAME: str = "Voice Ascend Webapp"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Paths (relative to project root)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    WEBAPP_DATA_DIR: Path = PROJECT_ROOT / "data" / "webapp"
    WEBAPP_AUDIO_DIR: Path = WEBAPP_DATA_DIR / "audio" / "recordings"
    WEBAPP_EXPORTS_DIR: Path = WEBAPP_DATA_DIR / "exports"
    WEBAPP_DB_DIR: Path = WEBAPP_DATA_DIR / "database"

    # ML Pipeline paths (共有参照用)
    ML_DATA_DIR: Path = PROJECT_ROOT / "data"
    ML_CACHE_DIR: Path = ML_DATA_DIR / "cache"
    ML_MODELS_DIR: Path = PROJECT_ROOT / "models"

    # Database
    DATABASE_URL: str = "sqlite:///data/webapp/database/webapp.db"

    # Audio settings
    MAX_AUDIO_DURATION: int = 30  # seconds
    MIN_AUDIO_DURATION: float = 0.5  # seconds
    TARGET_SAMPLE_RATE: int = 16000  # Hz (Whisper standard)
    ALLOWED_AUDIO_FORMATS: list = ["wav", "mp3", "webm", "ogg"]
    MAX_UPLOAD_SIZE_MB: int = 50

    # Language settings
    DEFAULT_LANGUAGE: str = "ja"  # 日本語

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# ディレクトリが存在することを確認
settings.WEBAPP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
settings.WEBAPP_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
settings.WEBAPP_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
