"""KB admin: CRUD документов базы знаний (UC7, роль admin)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field

from app.agent.factory import get_embedding_provider
from app.api.errors import ApiError
from app.core.auth import Principal, require_admin
from app.db.base import get_session, get_session_factory
from app.kb import service as kb
from app.kb.indexing import SessionFactory, index_in_background

router = APIRouter(prefix="/api/v1/kb/documents", tags=["kb"])

SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{1,126}$"


class DocumentIn(BaseModel):
    slug: str = Field(pattern=SLUG_PATTERN)
    title: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1, max_length=20_000)


class DocumentPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=512)
    content: str | None = Field(default=None, min_length=1, max_length=20_000)


def get_indexing_session_factory() -> SessionFactory:
    """Отдельная сессия для фоновой индексации: сессия запроса к тому моменту закрыта."""
    return get_session_factory()


def _translate(exc: kb.KbError) -> ApiError:
    return ApiError(exc.status_code, exc.code, str(exc))


@router.get("")
def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
    session=Depends(get_session),  # noqa: B008
    _: Principal = Depends(require_admin),  # noqa: B008
) -> dict:
    states, next_cursor = kb.list_documents(session, limit=limit, cursor=cursor)
    return {"items": [kb.serialize(state) for state in states], "next_cursor": next_cursor}


@router.get("/{document_id}")
def get_document(
    document_id: uuid.UUID,
    session=Depends(get_session),  # noqa: B008
    _: Principal = Depends(require_admin),  # noqa: B008
) -> dict:
    try:
        return kb.serialize(kb.get_document(session, document_id))
    except kb.KbError as exc:
        raise _translate(exc) from exc


@router.post("", status_code=201)
def create_document(
    body: DocumentIn,
    background: BackgroundTasks,
    session=Depends(get_session),  # noqa: B008
    session_factory: SessionFactory = Depends(get_indexing_session_factory),  # noqa: B008
    _: Principal = Depends(require_admin),  # noqa: B008
) -> dict:
    try:
        state = kb.create_document(session, slug=body.slug, title=body.title, content=body.content)
    except kb.KbError as exc:
        raise _translate(exc) from exc
    background.add_task(
        index_in_background, session_factory, state.latest.id, get_embedding_provider()
    )
    return kb.serialize(state)


@router.put("/{document_id}")
def update_document(
    document_id: uuid.UUID,
    body: DocumentPatch,
    background: BackgroundTasks,
    session=Depends(get_session),  # noqa: B008
    session_factory: SessionFactory = Depends(get_indexing_session_factory),  # noqa: B008
    _: Principal = Depends(require_admin),  # noqa: B008
) -> dict:
    try:
        state, reindex = kb.update_document(
            session, document_id, title=body.title, content=body.content
        )
    except kb.KbError as exc:
        raise _translate(exc) from exc
    if reindex:
        background.add_task(
            index_in_background, session_factory, state.latest.id, get_embedding_provider()
        )
    return kb.serialize(state)


@router.delete("/{document_id}")
def delete_document(
    document_id: uuid.UUID,
    session=Depends(get_session),  # noqa: B008
    _: Principal = Depends(require_admin),  # noqa: B008
) -> dict:
    try:
        document = kb.delete_document(session, document_id, now=datetime.now(UTC))
    except kb.KbError as exc:
        raise _translate(exc) from exc
    return {"id": str(document.id), "deleted_at": document.deleted_at.isoformat()}


@router.get("/{document_id}/versions")
def list_versions(
    document_id: uuid.UUID,
    session=Depends(get_session),  # noqa: B008
    _: Principal = Depends(require_admin),  # noqa: B008
) -> dict:
    try:
        rows = kb.versions(session, document_id)
    except kb.KbError as exc:
        raise _translate(exc) from exc
    return {
        "items": [
            {
                "id": str(row.id),
                "version": row.version,
                "content": row.content,
                "indexed": row.embedding_model is not None,
                "embedding_model": row.embedding_model,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }
