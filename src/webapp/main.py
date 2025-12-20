"""
FastAPI application entry point for Voice Ascend Whisper web data collection app.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
from loguru import logger

from webapp.config import get_settings
from webapp.database import create_db_and_tables, get_session
from webapp.routes import texts, recordings, audio, datasets


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Voice Ascend Whisper web application...")
    settings = get_settings()

    # Create necessary directories
    settings.WEBAPP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.WEBAPP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    settings.WEBAPP_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.WEBAPP_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    logger.info("Initializing database...")
    create_db_and_tables()
    logger.info("Database initialized successfully")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="Voice Ascend Whisper - Data Collection",
    description="Web application for collecting Japanese voice data for Whisper fine-tuning",
    version="0.1.0",
    lifespan=lifespan,
)

# Get settings
settings = get_settings()

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup Jinja2 templates
templates_path = Path(__file__).parent / "templates"
templates_path.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_path))

# Make templates available globally
app.state.templates = templates

# Include routers
app.include_router(texts.router, prefix="/texts", tags=["texts"])
app.include_router(recordings.router, prefix="/recordings", tags=["recordings"])
app.include_router(audio.router, prefix="/audio", tags=["audio"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])


@app.get("/")
async def root():
    """Root endpoint - redirect to index page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/index")


@app.get("/index")
async def index(
    request: __import__("fastapi").Request,
    session: Session = Depends(get_session),
):
    """Main dashboard page."""
    from webapp.services.text_service import TextService
    from webapp.services.recording_service import RecordingService
    from webapp.services.export_service import ExportService

    text_count = TextService.count_texts(session)
    recording_count = RecordingService.count_recordings(session)
    export_count = len(ExportService.get_exports(session))

    return app.state.templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "text_count": text_count,
            "recording_count": recording_count,
            "export_count": export_count,
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": "voice-ascend-whisper",
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "webapp.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
