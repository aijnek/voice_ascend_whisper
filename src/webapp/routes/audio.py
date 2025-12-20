"""
Audio file streaming API routes.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlmodel import Session

from webapp.config import get_settings, Settings
from webapp.database import get_session
from webapp.services.recording_service import RecordingService


router = APIRouter()


@router.get("/{recording_id}")
async def stream_audio(
    recording_id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """
    Stream audio file by recording ID.

    Args:
        recording_id: Recording ID

    Returns:
        FileResponse with audio file
    """
    # Get recording from database
    recording = RecordingService.get_recording(session=session, recording_id=recording_id)

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Get absolute file path
    file_path = RecordingService.get_recording_file_path(recording, settings)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Return audio file
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=recording.filename,
    )


@router.get("/file/{filename}")
async def stream_audio_by_filename(
    filename: str,
    settings: Settings = Depends(get_settings),
):
    """
    Stream audio file by filename (direct access).

    Args:
        filename: Audio filename

    Returns:
        FileResponse with audio file
    """
    # Construct file path
    file_path = settings.WEBAPP_AUDIO_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Security check: ensure file is within audio directory
    try:
        file_path.resolve().relative_to(settings.WEBAPP_AUDIO_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Return audio file
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )
