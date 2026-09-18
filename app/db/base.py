"""Подключение к Postgres - единственному source of truth (ADR-004)."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.singleton import once


@once
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        # pool_pre_ping: сессия не должна падать на первом же запросе после простоя.
        pool_pre_ping=True,
        # Размер пула задан явно: это потолок соединений процесса с Postgres.
        # Он меньше числа одновременных тикетов (NFR8) намеренно - соединение
        # не держится, пока агент ждёт LLM (см. release_connection).
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
    )


@once
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI-зависимость: одна сессия на запрос."""
    with get_session_factory()() as session:
        yield session


def release_connection(session: Session) -> None:
    """Вернуть соединение в пул перед долгим шагом без базы (вызов LLM).

    Сессия держит соединение от первого запроса до конца транзакции. Без этого
    вызова тикет держал бы соединение все секунды ожидания LLM, и число нужных
    соединений росло бы с числом одновременных тикетов, а не с числом
    одновременных обращений к базе.

    Применимо только к читающей транзакции: незаписанные изменения здесь - ошибка
    вызывающего кода, а не повод их молча закоммитить.
    """
    if session.new or session.dirty or session.deleted:
        raise RuntimeError("release_connection: в сессии есть незаписанные изменения")
    session.commit()
