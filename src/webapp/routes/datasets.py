"""Dataset export routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from sqlmodel import Session

from webapp.config import Settings, get_settings
from webapp.database import get_session
from webapp.models.dataset import DatasetExportCreate
from webapp.services.export_service import ExportService

router = APIRouter(tags=["datasets"])

templates = Jinja2Templates(directory="src/webapp/templates")


@router.get("/", response_class=HTMLResponse)
async def list_exports(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
):
    """Display list of dataset exports."""
    exports = ExportService.get_exports(session)

    return templates.TemplateResponse(
        "datasets/list.html",
        {
            "request": request,
            "exports": exports,
        },
    )


@router.get("/export", response_class=HTMLResponse)
async def export_page(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
):
    """Display export configuration page."""
    # Get statistics
    from webapp.services.recording_service import RecordingService

    total_recordings = RecordingService.count_recordings(session)
    validated_recordings = RecordingService.count_recordings(
        session, validated_only=True
    )

    return templates.TemplateResponse(
        "datasets/export.html",
        {
            "request": request,
            "total_recordings": total_recordings,
            "validated_recordings": validated_recordings,
        },
    )


@router.post("/export", response_class=HTMLResponse)
async def create_export(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    train_ratio: Annotated[float, Form()] = 80.0,
    dev_ratio: Annotated[float, Form()] = 10.0,
    test_ratio: Annotated[float, Form()] = 10.0,
    split_strategy: Annotated[str, Form()] = "random",
    min_duration: Annotated[float | None, Form()] = None,
    max_duration: Annotated[float | None, Form()] = None,
    validated_only: Annotated[bool, Form()] = False,
):
    """Create and execute dataset export."""
    try:
        # Validate ratios
        total_ratio = train_ratio + dev_ratio + test_ratio
        if abs(total_ratio - 100.0) > 0.1:
            raise HTTPException(
                status_code=400,
                detail=f"Split ratios must sum to 100% (got {total_ratio}%)",
            )

        # Create export configuration
        export_data = DatasetExportCreate(
            name=name,
            description=description,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            split_strategy=split_strategy,
            min_duration=min_duration,
            max_duration=max_duration,
            validated_only=validated_only,
        )

        # Create export entry
        db_export = ExportService.create_export(session, export_data, settings)
        logger.info(f"Created export {db_export.id}: {db_export.name}")

        # Execute export
        db_export = ExportService.execute_export(session, db_export.id, settings)
        logger.info(f"Export {db_export.id} completed successfully")

        # Return success result
        return templates.TemplateResponse(
            "datasets/export_result.html",
            {
                "request": request,
                "export": db_export,
                "success": True,
            },
        )

    except ValueError as e:
        logger.error(f"Export validation error: {e}")
        return templates.TemplateResponse(
            "datasets/export_result.html",
            {
                "request": request,
                "error": str(e),
                "success": False,
            },
        )
    except Exception as e:
        logger.exception(f"Export execution error: {e}")
        return templates.TemplateResponse(
            "datasets/export_result.html",
            {
                "request": request,
                "error": f"エクスポート中にエラーが発生しました: {str(e)}",
                "success": False,
            },
        )


@router.get("/{export_id}", response_class=HTMLResponse)
async def export_detail(
    request: Request,
    export_id: int,
    session: Annotated[Session, Depends(get_session)],
):
    """Display export details."""
    db_export = ExportService.get_export(session, export_id)
    if not db_export:
        raise HTTPException(status_code=404, detail="Export not found")

    return templates.TemplateResponse(
        "datasets/detail.html",
        {
            "request": request,
            "export": db_export,
        },
    )


@router.delete("/{export_id}", response_class=HTMLResponse)
async def delete_export(
    request: Request,
    export_id: int,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Delete an export."""
    deleted = ExportService.delete_export(session, export_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="Export not found")

    logger.info(f"Deleted export {export_id}")

    # Return empty response to remove the row from UI
    return HTMLResponse(content="", status_code=200)
