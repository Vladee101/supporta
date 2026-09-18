"""Outbox poller: переносит события из Postgres в RabbitMQ (ADR-007).

Один цикл - одна транзакция:

    BEGIN
      SELECT ... FROM outbox_events WHERE NOT published
        ORDER BY created_at LIMIT n FOR UPDATE SKIP LOCKED
      для каждого: publish + дождаться confirm → published = true
    COMMIT

`SKIP LOCKED` позволяет запускать несколько поллеров: строки, которые держит
один, другой просто пропускает, и событие не публикуется дважды параллельно.

Гарантия - **at-least-once**, не exactly-once. Если процесс упадёт после
confirm'а брокера, но до COMMIT, событие останется неопубликованным и уйдёт
повторно. Это закрывается на стороне consumer'а через inbox по
`idempotency_key` (ADR-007), а не здесь.

При первой же ошибке публикации цикл останавливается: сбой брокера обычно
общий, и долбить его остатком пачки бессмысленно. Чтобы одно «ядовитое»
событие не блокировало голову очереди навсегда, события с числом попыток
`MAX_ATTEMPTS` и больше из выборки исключаются - их показывает метрика
`stuck`, а разбирает runbook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import OutboxEvent
from app.messaging.publisher import EventPublisher

MAX_ATTEMPTS = 10
DEFAULT_BATCH_SIZE = 50


@dataclass(frozen=True, slots=True)
class PollResult:
    published: int
    failed: int
    stuck: int
    oldest_pending_age_seconds: float | None


def publish_pending(
    session: Session,
    publisher: EventPublisher,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    now: datetime | None = None,
) -> PollResult:
    now = now or datetime.now(UTC)

    events = session.scalars(
        select(OutboxEvent)
        .where(OutboxEvent.published.is_(False), OutboxEvent.attempts < MAX_ATTEMPTS)
        .order_by(OutboxEvent.created_at)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    ).all()

    published = failed = 0
    for event in events:
        try:
            publisher.publish(
                event.event_type,
                json.dumps(event.payload, ensure_ascii=False).encode("utf-8"),
                message_id=event.idempotency_key,
                priority=int(event.payload.get("priority", 0)),
            )
        except Exception as exc:  # noqa: BLE001 - любой сбой публикации фиксируется в строке
            event.attempts += 1
            event.last_error = f"{type(exc).__name__}: {exc}"[:1000]
            failed += 1
            break

        event.published = True
        event.published_at = now
        event.attempts += 1
        event.last_error = None
        published += 1

    session.commit()

    stuck = session.scalar(
        select(func.count())
        .select_from(OutboxEvent)
        .where(OutboxEvent.published.is_(False), OutboxEvent.attempts >= MAX_ATTEMPTS)
    )
    oldest = session.scalar(
        select(func.min(OutboxEvent.created_at)).where(OutboxEvent.published.is_(False))
    )
    # SLI из раздела «Наблюдаемость»: возраст самого старого неопубликованного
    # события. Больше пары минут - поллер встал или брокер недоступен.
    age = (now - oldest).total_seconds() if oldest is not None else None

    return PollResult(
        published=published,
        failed=failed,
        stuck=int(stuck or 0),
        oldest_pending_age_seconds=age,
    )
