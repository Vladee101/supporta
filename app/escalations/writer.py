"""Создание эскалации вместе с событием outbox.

Единственный способ создать эскалацию. Им пользуются и агент (решение Decision
Engine), и шедулер (таймаут уточнения, NFR9) - поэтому эскалация и событие для
брокера не могут разойтись ни в одном из путей: они пишутся в одной сессии, а
commit делает вызывающий код в той же транзакции, где меняется тикет
(ADR-004, ADR-007).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core import tracing
from app.db.models import Escalation, OutboxEvent, Ticket
from app.domain.enums import EscalationReason, EscalationStatus

ESCALATION_CREATED = "escalation.created"


def idempotency_key(escalation: Escalation) -> str:
    return f"{ESCALATION_CREATED}:{escalation.id}"


def create_escalation(
    session: Session,
    ticket: Ticket,
    *,
    reason: EscalationReason,
    rule_id: str,
    priority: int,
    category: str,
    draft_text: str | None = None,
    trace_id: str | None = None,
) -> Escalation:
    escalation = Escalation(
        ticket_id=ticket.id,
        reason=reason.value,
        rule_id=rule_id,
        status=EscalationStatus.PENDING.value,
        priority=priority,
        draft_text=draft_text,
    )
    session.add(escalation)
    session.flush()

    session.add(
        OutboxEvent(
            ticket_id=ticket.id,
            event_type=ESCALATION_CREATED,
            # Ключ дедупликации для consumer'а: outbox сам по себе не даёт
            # exactly-once, повторная публикация возможна (ADR-007).
            idempotency_key=idempotency_key(escalation),
            payload={
                "escalation_id": str(escalation.id),
                "ticket_id": str(ticket.id),
                "reason": reason.value,
                "rule_id": rule_id,
                "priority": priority,
                "category": category,
                "has_draft": draft_text is not None,
                # Сквозной trace_id: по нему событие в RabbitMQ и строки логов
                # consumer'а связываются с HTTP-запросом и записью audit_log.
                "trace_id": trace_id or tracing.ensure_trace_id(),
            },
        )
    )
    return escalation
