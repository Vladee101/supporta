"""Общие построители данных для тестов эскалаций."""

from __future__ import annotations

import uuid
from itertools import count

from sqlalchemy.orm import Session

from app.db.models import Escalation, Message, Operator, Ticket
from app.domain.enums import EscalationReason, MessageSender, OperatorRole, TicketStatus
from app.escalations.writer import create_escalation

_sequence = count(1)


def make_operator(session: Session, role: OperatorRole = OperatorRole.OPERATOR) -> Operator:
    number = next(_sequence)
    operator = Operator(
        name=f"Оператор {number}", email=f"op{number}-{uuid.uuid4().hex[:6]}@test", role=role
    )
    session.add(operator)
    session.flush()
    return operator


def make_escalated_ticket(
    session: Session,
    *,
    priority: int = 0,
    draft: str | None = "Черновик ответа агента.",
    reason: EscalationReason = EscalationReason.LOW_RAG_CONFIDENCE,
    status: TicketStatus = TicketStatus.ESCALATED_STANDARD,
    category: str = "faq",
) -> tuple[Ticket, Escalation]:
    ticket = Ticket(
        channel="web",
        external_id=f"ext-{uuid.uuid4().hex[:10]}",
        status=status.value,
        category=category,
        priority=priority,
    )
    session.add(ticket)
    session.flush()
    session.add(Message(ticket_id=ticket.id, sender=MessageSender.CLIENT, content="Вопрос клиента"))
    escalation = create_escalation(
        session,
        ticket,
        reason=reason,
        rule_id="R6",
        priority=priority,
        category=category,
        draft_text=draft,
    )
    session.flush()
    return ticket, escalation
