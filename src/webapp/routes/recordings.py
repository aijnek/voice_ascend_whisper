"""
Recording management API routes.
"""

from typing import Optional

from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlmodel import Session

from webapp.config import get_settings, Settings
from webapp.database import get_session
from webapp.services.recording_service import RecordingService
from webapp.services.text_service import TextService
from webapp.models.recording import Recording, RecordingCreate, RecordingUpdate


router = APIRouter(default_response_class=HTMLResponse)


def get_templates(request: Request):
    """Get Jinja2 templates from app state."""
    return request.app.state.templates


@router.get("/", response_class=HTMLResponse)
async def list_recordings(
    request: Request,
    text_id: Optional[int] = None,
    validated_only: bool = False,
    session: Session = Depends(get_session),
):
    """
    Display list of recordings.

    Query parameters:
    - text_id: Filter by text ID
    - validated_only: Only show validated recordings
    """
    templates = get_templates(request)

    # Get filtered recordings
    recordings = RecordingService.get_recordings(
        session=session,
        text_id=text_id,
        validated_only=validated_only,
    )

    return templates.TemplateResponse(
        "recordings/list.html",
        {
            "request": request,
            "recordings": recordings,
            "text_id": text_id,
            "validated_only": validated_only,
        }
    )


@router.get("/record", response_class=HTMLResponse)
async def record_page(
    request: Request,
    text_id: Optional[int] = None,
    show_all: bool = False,
    session: Session = Depends(get_session),
):
    """Display recording interface.

    Query parameters:
    - text_id: Pre-select specific text
    - show_all: Show all texts including recorded ones (default: False)
    """
    templates = get_templates(request)

    # Get texts without recordings (default) or all texts
    if show_all:
        texts = TextService.get_texts(session=session)
    else:
        texts = TextService.get_texts_without_recordings(session=session)

    # Get selected text if text_id is provided
    selected_text = None
    if text_id:
        selected_text = TextService.get_text(session=session, text_id=text_id)

    return templates.TemplateResponse(
        "recordings/record.html",
        {
            "request": request,
            "texts": texts,
            "selected_text": selected_text,
            "show_all": show_all,
        }
    )


@router.post("/", response_class=JSONResponse)
async def create_recording(
    text_id: int = Form(...),
    base64_audio: str = Form(...),
    notes: Optional[str] = Form(None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """
    Create a new recording from Base64 WAV audio.

    Form parameters:
    - text_id: Associated text ID
    - base64_audio: Base64-encoded WAV audio data
    - notes: Optional notes

    Returns:
        JSON response with recording details
    """
    try:
        # Verify text exists
        text = TextService.get_text(session=session, text_id=text_id)
        if not text:
            raise HTTPException(status_code=404, detail="Text not found")

        # Create recording data
        recording_data = RecordingCreate(
            text_id=text_id,
            notes=notes,
        )

        # Create recording with audio file
        new_recording = RecordingService.create_recording(
            session=session,
            recording_data=recording_data,
            base64_audio=base64_audio,
            settings=settings,
        )

        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": f"録音を保存しました（{new_recording.duration:.2f}秒）",
                "recording": {
                    "id": new_recording.id,
                    "text_id": new_recording.text_id,
                    "duration": new_recording.duration,
                    "filename": new_recording.filename,
                }
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recording failed: {str(e)}")


@router.get("/{recording_id}", response_class=HTMLResponse)
async def get_recording(
    request: Request,
    recording_id: int,
    session: Session = Depends(get_session),
):
    """Display single recording detail."""
    templates = get_templates(request)

    recording = RecordingService.get_recording(session=session, recording_id=recording_id)

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Get associated text
    text = TextService.get_text(session=session, text_id=recording.text_id)

    return templates.TemplateResponse(
        "recordings/detail.html",
        {
            "request": request,
            "recording": recording,
            "text": text,
        }
    )


@router.delete("/{recording_id}", response_class=HTMLResponse)
async def delete_recording(
    request: Request,
    recording_id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """Delete a recording."""
    templates = get_templates(request)

    success = RecordingService.delete_recording(
        session=session,
        recording_id=recording_id,
        settings=settings,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Return updated recording list
    recordings = RecordingService.get_recordings(session=session)

    return templates.TemplateResponse(
        "recordings/list.html",
        {
            "request": request,
            "recordings": recordings,
            "message": "録音を削除しました",
        }
    )


@router.put("/{recording_id}/validate", response_class=HTMLResponse)
async def validate_recording(
    request: Request,
    recording_id: int,
    is_validated: bool = Form(...),
    notes: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Validate or invalidate a recording."""
    templates = get_templates(request)

    updated_recording = RecordingService.validate_recording(
        session=session,
        recording_id=recording_id,
        is_valid=is_validated,
        notes=notes,
    )

    if not updated_recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Return updated recording list
    recordings = RecordingService.get_recordings(session=session)

    status_text = "検証済み" if is_validated else "未検証"
    return templates.TemplateResponse(
        "recordings/list.html",
        {
            "request": request,
            "recordings": recordings,
            "message": f"録音を{status_text}にマークしました",
        }
    )
