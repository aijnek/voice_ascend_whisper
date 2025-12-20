"""
Text management API routes.
"""

from typing import Optional

from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlmodel import Session

from webapp.database import get_session
from webapp.services.text_service import TextService
from webapp.models.text import Text, TextCreate, TextUpdate


router = APIRouter(default_response_class=HTMLResponse)


def get_templates(request: Request):
    """Get Jinja2 templates from app state."""
    return request.app.state.templates


@router.get("/", response_class=HTMLResponse)
async def list_texts(
    request: Request,
    language: Optional[str] = None,
    source: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """
    Display list of texts.

    Query parameters:
    - language: Filter by language code (e.g., 'ja')
    - source: Filter by source ('manual', 'llm_generated', 'imported')
    """
    templates = get_templates(request)

    # Get filtered texts
    texts = TextService.get_texts(
        session=session,
        language=language,
        source=source,
    )

    return templates.TemplateResponse(
        "texts/list.html",
        {
            "request": request,
            "texts": texts,
            "language": language,
            "source": source,
        }
    )


@router.get("/new", response_class=HTMLResponse)
async def new_text_form(request: Request):
    """Display form for creating new text."""
    templates = get_templates(request)

    return templates.TemplateResponse(
        "texts/form.html",
        {
            "request": request,
            "text": None,
            "mode": "create",
        }
    )


@router.post("/", response_class=HTMLResponse)
async def create_text(
    request: Request,
    content: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    source: str = Form("manual"),
    language: str = Form("ja"),
    session: Session = Depends(get_session),
):
    """
    Create a new text entry.

    Returns the updated text list (HTMX partial).
    """
    templates = get_templates(request)

    try:
        # Form()で受け取ったデータをPydanticモデルに変換
        text_data = TextCreate(
            content=content,
            description=description,
            tags=tags,
            source=source,
            language=language,
        )

        # サービス層でPydanticモデルを使用
        new_text = TextService.create_text(
            session=session,
            text_data=text_data,
        )

        # Return updated text list
        texts = TextService.get_texts(session=session)

        return templates.TemplateResponse(
            "texts/list.html",
            {
                "request": request,
                "texts": texts,
                "message": f"テキスト「{new_text.content[:20]}...」を追加しました",
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{text_id}", response_class=HTMLResponse)
async def get_text(
    request: Request,
    text_id: int,
    session: Session = Depends(get_session),
):
    """Display single text detail."""
    templates = get_templates(request)

    text = TextService.get_text(session=session, text_id=text_id)

    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    return templates.TemplateResponse(
        "texts/detail.html",
        {
            "request": request,
            "text": text,
        }
    )


@router.get("/{text_id}/edit", response_class=HTMLResponse)
async def edit_text_form(
    request: Request,
    text_id: int,
    session: Session = Depends(get_session),
):
    """Display form for editing text."""
    templates = get_templates(request)

    text = TextService.get_text(session=session, text_id=text_id)

    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    return templates.TemplateResponse(
        "texts/form.html",
        {
            "request": request,
            "text": text,
            "mode": "edit",
        }
    )


@router.put("/{text_id}", response_class=HTMLResponse)
async def update_text(
    request: Request,
    text_id: int,
    content: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Update an existing text entry."""
    templates = get_templates(request)

    try:
        # Form()で受け取ったデータをPydanticモデルに変換
        text_data = TextUpdate(
            content=content,
            description=description,
            tags=tags,
        )

        updated_text = TextService.update_text(
            session=session,
            text_id=text_id,
            text_data=text_data,
        )

        if not updated_text:
            raise HTTPException(status_code=404, detail="Text not found")

        # Return updated text list
        texts = TextService.get_texts(session=session)

        return templates.TemplateResponse(
            "texts/list.html",
            {
                "request": request,
                "texts": texts,
                "message": f"テキストを更新しました",
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{text_id}", response_class=HTMLResponse)
async def delete_text(
    request: Request,
    text_id: int,
    session: Session = Depends(get_session),
):
    """Delete a text entry."""
    templates = get_templates(request)

    success = TextService.delete_text(session=session, text_id=text_id)

    if not success:
        raise HTTPException(status_code=404, detail="Text not found")

    # Return updated text list
    texts = TextService.get_texts(session=session)

    return templates.TemplateResponse(
        "texts/list.html",
        {
            "request": request,
            "texts": texts,
            "message": "テキストを削除しました",
        }
    )
