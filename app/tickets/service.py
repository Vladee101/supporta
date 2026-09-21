"""Клиентская сторона тикета: приём (UC1), ответ на уточнение, запрос человека (UC9).

Агент здесь не вызывается - только состояние тикета. Запуск пайплайна - забота
роутера: так правила «когда агент вообще работает» видны в одном месте и
тестируются без классификатора и RAG.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Escalation, Message, Ticket
from app.domain.decision import DecisionInput, decide
from app.domain.enums import (
    Category,
    EscalationStatus,
    MessageSender,
    TicketStatus,
)
from app.escalations.writer import create_escalation


class TicketError(Exception):
    code = "ticket_error"
    status_code = 400


class TicketNotFoundError(TicketError):
    code = "not_found"
    status_code = 404


class TicketClosedError(TicketError):
    code = "ticket_closed"
    status_code = 409


class EscalationWindowExpiredError(TicketError):
    code = "escalation_window_expired"
    status_code = 409


#: Состояния, в которых новое сообщение клиента не запускает агента: тикет уже
#: у человека или ждёт его, сообщение просто ложится в историю для оператора.
_WITH_HUMAN = {
    TicketStatus.ESCALATED_STANDARD.value,
    TicketStatus.ESCALATED_PRIORITY.value,
    TicketStatus.PENDING_OPERATOR.value,
    TicketStatus.IN_PROGRESS.value,
}
_CLOSED = {TicketStatus.RESOLVED_AUTO.value, TicketStatus.RESOLVED_BY_OPERATOR.value}


@dataclass(frozen=True, slots=True)
class Intake:
    ticket: Ticket
    created: bool


def get_ticket(session: Session, ticket_id: uuid.UUID) -> Ticket:
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise TicketNotFoundError("тикет не найден")
    return ticket


def create_ticket(
    session: Session,
    *,
    channel: str,
    external_id: str,
    content: str,
    client_id: str | None = None,
) -> Intake:
    """Идемпотентно по `(channel, external_id)`: повтор вебхука не создаёт дубль."""
    existing = session.scalar(
        select(Ticket).where(Ticket.channel == channel, Ticket.external_id == external_id)
    )
    if existing is not None:
        return Intake(ticket=existing, created=False)

    ticket = Ticket(
        channel=channel,
        external_id=external_id,
        client_id=client_id,
        status=TicketStatus.NEW.value,
    )
    try:
        with session.begin_nested():
            session.add(ticket)
            session.flush()
    except IntegrityError:
        # Параллельная доставка того же вебхука успела раньше - отдаём её тикет.
        existing = session.scalar(
            select(Ticket).where(Ticket.channel == channel, Ticket.external_id == external_id)
        )
        return Intake(ticket=existing, created=False)

    session.add(Message(ticket_id=ticket.id, sender=MessageSender.CLIENT, content=content))
    session.commit()
    return Intake(ticket=ticket, created=True)


def add_client_message(session: Session, ticket: Ticket, content: str, *, now: datetime) -> bool:
    """Добавить сообщение клиента. Возвращает True, если нужно запустить агента."""
    if ticket.status in _CLOSED:
        raise TicketClosedError(
            "тикет закрыт; если ответ не помог, запросите оператора (POST /escalate)"
        )

    session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.CLIENT,
            iteration=ticket.clarification_count,
            content=content,
        )
    )
    ticket.last_client_reply_at = now
    session.commit()
    # Агент работает только на ответе на уточнение. Если тикет уже у человека,
    # повторный прогон агента мог бы, например, автоответить поверх оператора.
    return ticket.status == TicketStatus.AWAITING_CLARIFICATION.value


def request_human(
    session: Session, ticket: Ticket, *, now: datetime, window: timedelta
) -> Escalation:
    """UC9: явный запрос оператора. Маршрут всё равно выбирает Decision Engine (R1)."""
    if ticket.status == TicketStatus.RESOLVED_BY_OPERATOR.value:
        raise TicketClosedError("тикет уже закрыт оператором")
    if ticket.status == TicketStatus.RESOLVED_AUTO.value and (
        ticket.resolved_at is None or now - ticket.resolved_at > window
    ):
        raise EscalationWindowExpiredError(
            "окно эскалации после автоответа истекло - создайте новое обращение"
        )

    decision = decide(
        DecisionInput(Category(ticket.category or Category.UNCLASSIFIED), human_requested=True)
    )

    open_escalation = session.scalar(
        select(Escalation).where(
            Escalation.ticket_id == ticket.id,
            Escalation.status != EscalationStatus.RESOLVED.value,
        )
    )
    if open_escalation is not None:
        # Тикет уже ждёт человека: вторую эскалацию не создаём, поднимаем приоритет
        # существующей - очередь оператора (источник истины) пересортируется сама.
        open_escalation.priority = max(open_escalation.priority, decision.priority)
        escalation = open_escalation
        if ticket.status == TicketStatus.ESCALATED_STANDARD.value:
            ticket.status = TicketStatus.ESCALATED_PRIORITY.value
    else:
        escalation = create_escalation(
            session,
            ticket,
            reason=decision.reason,
            rule_id=decision.rule_id,
            priority=decision.priority,
            category=ticket.category or Category.UNCLASSIFIED.value,
        )
        ticket.status = TicketStatus.ESCALATED_PRIORITY.value
        ticket.resolved_at = None

    ticket.priority = max(ticket.priority, decision.priority)
    session.add(
        AuditLog(
            ticket_id=ticket.id,
            actor="client",
            action=decision.action.value,
            rule_id=decision.rule_id,
            payload={
                "reason": decision.reason.value,
                "escalation_id": str(escalation.id),
                "reused_open_escalation": open_escalation is not None,
            },
        )
    )
    session.commit()
    return escalation


def public_view(session: Session, ticket: Ticket) -> dict:
    """Что видит клиент: статус и переписку. Классификация и трейс - внутренние."""
    messages = session.scalars(
        select(Message).where(Message.ticket_id == ticket.id).order_by(Message.created_at)
    ).all()
    return {
        "ticket_id": str(ticket.id),
        "status": ticket.status,
        "messages": [
            {
                "sender": message.sender,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in messages
        ],
    }
