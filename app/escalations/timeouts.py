"""Принудительная эскалация по таймауту ответа на уточнение (NFR9, FR11).

Тикет в `awaiting_clarification`, по которому клиент молчит дольше таймаута,
эскалируется по R7b с причиной `clarification_timeout`. Эскалация создаётся
тем же `create_escalation`, что и у агента, - вместе с событием outbox и в
одной транзакции с изменением тикета.

Точка отсчёта - последнее сообщение агента (сам запрос уточнения), а не
создание тикета: иначе тикет, долго ждавший классификации, эскалировался бы
раньше, чем клиент вообще увидел вопрос.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Message, Ticket
from app.domain.decision import PRIORITY_STANDARD
from app.domain.enums import EscalationReason, MessageSender, TicketStatus
from app.escalations.writer import create_escalation


def escalate_clarification_timeouts(session: Session, *, now: datetime, timeout: timedelta) -> int:
    last_agent_message = (
        select(Message.ticket_id, func.max(Message.created_at).label("asked_at"))
        .where(Message.sender == MessageSender.AGENT)
        .group_by(Message.ticket_id)
        .subquery()
    )

    tickets = session.scalars(
        select(Ticket)
        .join(last_agent_message, last_agent_message.c.ticket_id == Ticket.id)
        .where(
            Ticket.status == TicketStatus.AWAITING_CLARIFICATION.value,
            last_agent_message.c.asked_at < now - timeout,
        )
        .with_for_update(of=Ticket, skip_locked=True)
    ).all()

    for ticket in tickets:
        escalation = create_escalation(
            session,
            ticket,
            reason=EscalationReason.CLARIFICATION_TIMEOUT,
            rule_id="R7b",
            priority=PRIORITY_STANDARD,
            category=ticket.category or "unclassified",
        )
        ticket.status = TicketStatus.ESCALATED_STANDARD.value
        session.add(
            AuditLog(
                ticket_id=ticket.id,
                actor="scheduler",
                action="A2",
                rule_id="R7b",
                payload={
                    "reason": EscalationReason.CLARIFICATION_TIMEOUT.value,
                    "escalation_id": str(escalation.id),
                    "timeout_minutes": int(timeout.total_seconds() // 60),
                },
            )
        )

    session.commit()
    return len(tickets)
