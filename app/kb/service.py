"""Управление базой знаний (UC7, ADR-011).

Две разные «версии» документа, которые важно не путать:

* **последняя** (`latest`) - самая свежая правка администратора;
* **текущая** (`current_version_id`) - та, что участвует в RAG.

Правка создаёт новую последнюю версию без embedding'а, а текущей она
становится только после индексации (`app.kb.indexing`). До этого момента
агент продолжает отвечать по предыдущей версии: документ не пропадает из
выдачи на время переиндексации, а сломанная индексация не выключает его.
Новый документ, у которого текущей версии ещё нет, в RAG не участвует - это
и есть «доступен без деплоя, но после индексации» из AC UC7.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import KbDocument, KbDocumentVersion


class KbError(Exception):
    code = "kb_error"
    status_code = 400


class DocumentNotFoundError(KbError):
    code = "not_found"
    status_code = 404


class SlugTakenError(KbError):
    code = "slug_taken"
    status_code = 409


@dataclass(frozen=True, slots=True)
class DocumentState:
    document: KbDocument
    latest: KbDocumentVersion
    current: KbDocumentVersion | None

    @property
    def indexing(self) -> bool:
        return self.current is None or self.current.id != self.latest.id


def _latest_version(session: Session, document_id: uuid.UUID) -> KbDocumentVersion:
    return session.scalar(
        select(KbDocumentVersion)
        .where(KbDocumentVersion.document_id == document_id)
        .order_by(KbDocumentVersion.version.desc())
        .limit(1)
    )


def _state(session: Session, document: KbDocument) -> DocumentState:
    current = (
        session.get(KbDocumentVersion, document.current_version_id)
        if document.current_version_id
        else None
    )
    return DocumentState(
        document=document, latest=_latest_version(session, document.id), current=current
    )


def _load(session: Session, document_id: uuid.UUID, *, include_deleted: bool = False) -> KbDocument:
    document = session.get(KbDocument, document_id)
    if document is None or (document.deleted_at is not None and not include_deleted):
        raise DocumentNotFoundError("документ не найден")
    return document


def list_documents(
    session: Session, *, limit: int = 50, cursor: str | None = None
) -> tuple[list[DocumentState], str | None]:
    """Неудалённые документы по slug; курсор - последний slug страницы."""
    stmt = select(KbDocument).where(KbDocument.deleted_at.is_(None))
    if cursor:
        stmt = stmt.where(KbDocument.slug > cursor)
    rows = session.scalars(stmt.order_by(KbDocument.slug).limit(limit + 1)).all()
    page = rows[:limit]
    next_cursor = page[-1].slug if len(rows) > limit else None
    return [_state(session, document) for document in page], next_cursor


def get_document(session: Session, document_id: uuid.UUID) -> DocumentState:
    return _state(session, _load(session, document_id))


def create_document(session: Session, *, slug: str, title: str, content: str) -> DocumentState:
    document = KbDocument(slug=slug, title=title)
    try:
        with session.begin_nested():
            session.add(document)
            session.flush()
    except IntegrityError as exc:
        raise SlugTakenError(f"slug {slug!r} уже занят") from exc

    session.add(KbDocumentVersion(document_id=document.id, version=1, content=content))
    session.commit()
    return _state(session, document)


def update_document(
    session: Session,
    document_id: uuid.UUID,
    *,
    title: str | None = None,
    content: str | None = None,
) -> tuple[DocumentState, bool]:
    """Возвращает состояние и признак «создана новая версия, нужна индексация»."""
    document = _load(session, document_id)
    latest = _latest_version(session, document.id)

    if title is not None:
        document.title = title

    new_version = content is not None and content != latest.content
    if new_version:
        session.add(
            KbDocumentVersion(document_id=document.id, version=latest.version + 1, content=content)
        )
    session.commit()
    return _state(session, document), new_version


def delete_document(session: Session, document_id: uuid.UUID, *, now: datetime) -> KbDocument:
    """Soft-delete: документ уходит из RAG, версии и история ретривалов остаются (UC7)."""
    document = _load(session, document_id, include_deleted=True)
    if document.deleted_at is None:
        document.deleted_at = now
        session.commit()
    return document


def versions(session: Session, document_id: uuid.UUID) -> list[KbDocumentVersion]:
    _load(session, document_id, include_deleted=True)
    return list(
        session.scalars(
            select(KbDocumentVersion)
            .where(KbDocumentVersion.document_id == document_id)
            .order_by(KbDocumentVersion.version)
        )
    )


def serialize(state: DocumentState) -> dict:
    document = state.document
    return {
        "id": str(document.id),
        "slug": document.slug,
        "title": document.title,
        "content": state.latest.content,
        "latest_version": state.latest.version,
        "current_version": state.current.version if state.current else None,
        # Правка ещё не проиндексирована: агент пока отвечает по current_version.
        "indexing": state.indexing,
        "embedding_model": state.current.embedding_model if state.current else None,
        "updated_at": document.updated_at.isoformat(),
    }
