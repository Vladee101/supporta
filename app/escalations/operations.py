"""Операции оператора над эскалацией: очередь, claim, release, resolve (UC6, FR10).

Claim сделан одним условным UPDATE, а не парой SELECT + UPDATE:

    UPDATE escalations SET locked_by = :me, locked_until = :now + ttl, ...
     WHERE id = :id AND status <> 'resolved'
       AND (locked_by IS NULL OR locked_until < :now OR locked_by = :me)
    RETURNING id

Два оператора, нажавшие «взять» одновременно, выполняют один и тот же UPDATE;
Postgres сериализует их на блокировке строки, и условие WHERE совпадёт только
у первого. Проверка «свободно ли» и захват неразделимы - гонки между ними нет.

Истёкший claim считается свободным сразу, без ожидания шедулера (ленивое
истечение): шедулер лишь приводит статусы в порядок для отображения очереди.
"""

from __future__ import annotations

import base64
import difflib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from app.db.models import (
    AuditLog,
    Classification,
    Escalation,
    Message,
    OperatorAction,
    RagRetrieval,
    Ticket,
)
from app.domain.enums import (
    EscalationStatus,
    MessageSender,
    OperatorActionType,
    TicketStatus,
)


class EscalationError(Exception):
    code = "escalation_error"
    status_code = 400


class EscalationNotFoundError(EscalationError):
    code = "not_found"
    status_code = 404


class AlreadyLockedError(EscalationError):
    code = "already_locked"
    status_code = 409


class AlreadyResolvedError(EscalationError):
    code = "already_resolved"
    status_code = 409


class NotClaimedError(EscalationError):
    code = "not_claimed"
    status_code = 409


class InvalidResolutionError(EscalationError):
    code = "invalid_resolution"
    status_code = 422


def _available_condition(now: datetime):
    """Эскалация ждёт оператора: новая или с истёкшим claim'ом."""
    return or_(
        Escalation.status == EscalationStatus.PENDING.value,
        and_(
            Escalation.status == EscalationStatus.IN_PROGRESS.value,
            Escalation.locked_until < now,
        ),
    )


# ---------------------------------------------------------------------------
# Очередь
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueuePage:
    items: list[Escalation]
    next_cursor: str | None


def _encode_cursor(escalation: Escalation) -> str:
    raw = json.dumps(
        [escalation.priority, escalation.created_at.isoformat(), str(escalation.id)]
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> tuple[int, datetime, uuid.UUID]:
    try:
        priority, created_at, escalation_id = json.loads(base64.urlsafe_b64decode(cursor))
        return int(priority), datetime.fromisoformat(created_at), uuid.UUID(escalation_id)
    except (ValueError, TypeError) as exc:
        raise EscalationError(f"некорректный cursor: {cursor!r}") from exc


def queue(
    session: Session, *, now: datetime, limit: int = 20, cursor: str | None = None
) -> QueuePage:
    """Очередь по `priority DESC, created_at ASC` - keyset-пагинация.

    Keyset, а не OFFSET: очередь меняется, пока оператор листает, и OFFSET
    пропускал бы или дублировал записи на границе страниц.
    """
    stmt = select(Escalation).where(_available_condition(now))

    if cursor:
        priority, created_at, escalation_id = _decode_cursor(cursor)
        stmt = stmt.where(
            or_(
                Escalation.priority < priority,
                and_(Escalation.priority == priority, Escalation.created_at > created_at),
                and_(
                    Escalation.priority == priority,
                    Escalation.created_at == created_at,
                    Escalation.id > escalation_id,
                ),
            )
        )

    rows = session.scalars(
        stmt.order_by(
            Escalation.priority.desc(), Escalation.created_at.asc(), Escalation.id.asc()
        ).limit(limit + 1)
    ).all()

    items = list(rows[:limit])
    next_cursor = _encode_cursor(items[-1]) if len(rows) > limit else None
    return QueuePage(items=items, next_cursor=next_cursor)


# ---------------------------------------------------------------------------
# Claim / release
# ---------------------------------------------------------------------------


def _load(session: Session, escalation_id: uuid.UUID) -> Escalation:
    escalation = session.get(Escalation, escalation_id, populate_existing=True)
    if escalation is None:
        raise EscalationNotFoundError(f"эскалация {escalation_id} не найдена")
    return escalation


def _audit(session: Session, ticket_id: uuid.UUID, action: str, payload: dict) -> None:
    session.add(AuditLog(ticket_id=ticket_id, actor="operator", action=action, payload=payload))


def claim(
    session: Session,
    escalation_id: uuid.UUID,
    operator_id: uuid.UUID,
    *,
    now: datetime,
    ttl: timedelta,
) -> Escalation:
    claimed = session.execute(
        update(Escalation)
        .where(
            Escalation.id == escalation_id,
            Escalation.status != EscalationStatus.RESOLVED.value,
            or_(
                Escalation.locked_by.is_(None),
                Escalation.locked_until < now,
                Escalation.locked_by == operator_id,
            ),
        )
        .values(
            locked_by=operator_id,
            locked_until=now + ttl,
            operator_id=operator_id,
            status=EscalationStatus.IN_PROGRESS.value,
        )
        .returning(Escalation.id)
        .execution_options(synchronize_session=False)
    ).first()

    if claimed is None:
        # Несовпавший UPDATE ничего не изменил - откатывать нечего.
        escalation = _load(session, escalation_id)
        if escalation.status == EscalationStatus.RESOLVED.value:
            raise AlreadyResolvedError("эскалация уже закрыта")
        raise AlreadyLockedError(
            f"эскалация в работе у другого оператора до {escalation.locked_until.isoformat()}"
        )

    escalation = _load(session, escalation_id)
    ticket = session.get(Ticket, escalation.ticket_id)
    ticket.status = TicketStatus.IN_PROGRESS.value
    _audit(
        session,
        ticket.id,
        "claim",
        {
            "escalation_id": str(escalation.id),
            "operator_id": str(operator_id),
            "locked_until": escalation.locked_until.isoformat(),
        },
    )
    session.commit()
    return escalation


def release(
    session: Session, escalation_id: uuid.UUID, operator_id: uuid.UUID, *, now: datetime
) -> Escalation:
    escalation = _load(session, escalation_id)
    if escalation.status == EscalationStatus.RESOLVED.value:
        raise AlreadyResolvedError("эскалация уже закрыта")
    if escalation.locked_by != operator_id:
        raise NotClaimedError("освободить эскалацию может только оператор, который её взял")

    _return_to_queue(session, escalation)
    _audit(
        session,
        escalation.ticket_id,
        "release",
        {"escalation_id": str(escalation.id), "operator_id": str(operator_id)},
    )
    session.commit()
    return escalation


def _return_to_queue(session: Session, escalation: Escalation) -> None:
    escalation.status = EscalationStatus.PENDING.value
    escalation.locked_by = None
    escalation.locked_until = None
    ticket = session.get(Ticket, escalation.ticket_id)
    if ticket.status == TicketStatus.IN_PROGRESS.value:
        ticket.status = TicketStatus.PENDING_OPERATOR.value


def release_expired(session: Session, *, now: datetime) -> int:
    """Вернуть в очередь эскалации с истёкшим claim'ом (для шедулера)."""
    expired = session.scalars(
        select(Escalation)
        .where(
            Escalation.status == EscalationStatus.IN_PROGRESS.value,
            Escalation.locked_until < now,
        )
        .with_for_update(skip_locked=True)
    ).all()
    for escalation in expired:
        previous_owner = escalation.locked_by
        _return_to_queue(session, escalation)
        session.add(
            AuditLog(
                ticket_id=escalation.ticket_id,
                actor="scheduler",
                action="claim_expired",
                payload={
                    "escalation_id": str(escalation.id),
                    "operator_id": str(previous_owner) if previous_owner else None,
                },
            )
        )
    session.commit()
    return len(expired)


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def resolve(
    session: Session,
    escalation_id: uuid.UUID,
    operator_id: uuid.UUID,
    *,
    action: OperatorActionType,
    final_text: str | None,
    now: datetime,
) -> OperatorAction:
    escalation = session.scalar(
        select(Escalation)
        .where(Escalation.id == escalation_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if escalation is None:
        raise EscalationNotFoundError(f"эскалация {escalation_id} не найдена")
    if escalation.status == EscalationStatus.RESOLVED.value:
        raise AlreadyResolvedError("эскалация уже закрыта")
    if escalation.locked_by != operator_id or escalation.locked_until < now:
        raise NotClaimedError("перед ответом эскалацию нужно взять в работу (claim)")

    text = _final_text(escalation, action, final_text)

    operator_action = OperatorAction(
        escalation_id=escalation.id,
        operator_id=operator_id,
        action_type=action.value,
        draft_text=escalation.draft_text,
        final_text=text,
    )
    session.add(operator_action)

    escalation.status = EscalationStatus.RESOLVED.value
    escalation.resolved_at = now
    escalation.locked_by = None
    escalation.locked_until = None

    ticket = session.get(Ticket, escalation.ticket_id)
    ticket.status = TicketStatus.RESOLVED_BY_OPERATOR.value
    ticket.resolved_at = now

    session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.OPERATOR,
            iteration=ticket.clarification_count,
            content=text,
        )
    )
    _audit(
        session,
        ticket.id,
        f"resolve:{action.value}",
        {
            "escalation_id": str(escalation.id),
            "operator_id": str(operator_id),
            "had_draft": escalation.draft_text is not None,
            # Сходство черновика и финального ответа - сырьё для метрики
            # operator override rate (NFR7, раздел «Наблюдаемость»).
            "draft_similarity": _similarity(escalation.draft_text, text),
        },
    )
    session.commit()
    return operator_action


def _final_text(
    escalation: Escalation, action: OperatorActionType, final_text: str | None
) -> str:
    cleaned = (final_text or "").strip()

    if action is OperatorActionType.CONFIRM:
        if escalation.draft_text is None:
            # ADR-008: у жалоб и возвратов черновика нет - подтверждать нечего.
            raise InvalidResolutionError("у эскалации нет черновика: используйте edit или reject")
        if cleaned and cleaned != escalation.draft_text.strip():
            raise InvalidResolutionError("confirm отправляет черновик как есть; для правок - edit")
        return escalation.draft_text

    if not cleaned:
        raise InvalidResolutionError(f"для действия {action.value} нужен final_text")
    return cleaned


def _similarity(draft: str | None, final: str) -> float | None:
    if draft is None:
        return None
    return round(difflib.SequenceMatcher(a=draft, b=final).ratio(), 4)


# ---------------------------------------------------------------------------
# Пакет контекста для оператора (UC5)
# ---------------------------------------------------------------------------


def context(session: Session, escalation_id: uuid.UUID) -> dict:
    escalation = _load(session, escalation_id)
    ticket = session.get(Ticket, escalation.ticket_id)

    messages = session.scalars(
        select(Message).where(Message.ticket_id == ticket.id).order_by(Message.created_at)
    ).all()
    classifications = session.scalars(
        select(Classification)
        .where(Classification.ticket_id == ticket.id)
        .order_by(Classification.iteration)
    ).all()
    retrievals = session.scalars(
        select(RagRetrieval)
        .where(RagRetrieval.ticket_id == ticket.id)
        .order_by(RagRetrieval.iteration, RagRetrieval.rank)
    ).all()

    return {
        "escalation": serialize_escalation(escalation),
        "ticket": {
            "id": str(ticket.id),
            "channel": ticket.channel,
            "status": ticket.status,
            "category": ticket.category,
            "risk_level": ticket.risk_level,
            "clarification_count": ticket.clarification_count,
            "created_at": ticket.created_at.isoformat(),
        },
        # Оператору показываем оригинал: он уполномочен видеть PII клиента,
        # маскирование защищает LLM и логи, а не человека, который отвечает.
        "messages": [
            {
                "sender": message.sender,
                "iteration": message.iteration,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in messages
        ],
        "classifications": [
            {
                "iteration": row.iteration,
                "category": row.category,
                "confidence": row.confidence,
                "confidence_source": row.confidence_source,
                "model_id": row.model_id,
                "reasoning": row.reasoning,
            }
            for row in classifications
        ],
        "documents": [
            {
                "iteration": row.iteration,
                "rank": row.rank,
                "relevance_score": row.relevance_score,
                "document_version_id": str(row.document_version_id),
                "snapshot": row.chunk_snapshot,
            }
            for row in retrievals
        ],
    }


def serialize_escalation(escalation: Escalation) -> dict:
    return {
        "id": str(escalation.id),
        "ticket_id": str(escalation.ticket_id),
        "reason": escalation.reason,
        "rule_id": escalation.rule_id,
        "status": escalation.status,
        "priority": escalation.priority,
        "draft_text": escalation.draft_text,
        "locked_by": str(escalation.locked_by) if escalation.locked_by else None,
        "locked_until": escalation.locked_until.isoformat() if escalation.locked_until else None,
        "created_at": escalation.created_at.isoformat(),
        "resolved_at": escalation.resolved_at.isoformat() if escalation.resolved_at else None,
    }
