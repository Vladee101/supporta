"""Индексация версии документа и её активация.

Указатель `current_version_id` двигается только вперёд. Две быстрые правки
подряд порождают две фоновые индексации, и завершиться они могут в любом
порядке; без проверки номера версии более старая правка, проиндексированная
позже, вернула бы документ к устаревшему тексту.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KbDocument, KbDocumentVersion
from app.services.embeddings import EmbeddingProvider

log = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractContextManager[Session]]


def index_version(session: Session, version_id: uuid.UUID, provider: EmbeddingProvider) -> bool:
    """Посчитать embedding и активировать версию. True - версия стала текущей."""
    version = session.get(KbDocumentVersion, version_id)
    if version is None:
        return False

    if version.embedding is None or version.embedding_model != provider.model_id:
        version.embedding = provider.encode([version.content])[0]
        version.embedding_model = provider.model_id

    # Блокировка документа сериализует конкурирующие активации его версий.
    document = session.scalar(
        select(KbDocument).where(KbDocument.id == version.document_id).with_for_update()
    )
    current = (
        session.get(KbDocumentVersion, document.current_version_id)
        if document.current_version_id
        else None
    )
    activated = current is None or version.version > current.version
    if activated:
        document.current_version_id = version.id
    session.commit()
    return activated


def index_in_background(
    session_factory: SessionFactory, version_id: uuid.UUID, provider: EmbeddingProvider
) -> None:
    """Точка входа для BackgroundTasks: своя сессия, ошибки - в лог, не в ответ."""
    try:
        with session_factory() as session:
            activated = index_version(session, version_id, provider)
        log.info("версия %s проиндексирована, активирована: %s", version_id, activated)
    except Exception:
        # Документ остаётся на предыдущей версии - агент продолжает отвечать,
        # а повторить индексацию можно scripts.index_kb.
        log.exception("индексация версии %s не удалась", version_id)
