"""Retention (NFR4): персональные данные вычищаются, аудит и метрики остаются."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.db.models import (
    AuditLog,
    Classification,
    Escalation,
    Message,
    OperatorAction,
    Ticket,
)
from app.domain.enums import MessageSender, TicketStatus
from app.retention import TOMBSTONE, scrub_expired
from tests.escalation_fixtures import make_operator

pytestmark = pytest.mark.integration

RETENTION = timedelta(days=90)
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
PHONE = "+7 999 123-45-67"
NAME = "Иван Петров"


def make_ticket(session, *, age_days: int, status: TicketStatus, with_operator_reply: bool = False):
    ticket = Ticket(
        channel="web",
        external_id=f"ret-{uuid.uuid4().hex[:10]}",
        client_id="client-42",
        status=status.value,
        category="faq",
        created_at=NOW - timedelta(days=age_days),
    )
    session.add(ticket)
    session.flush()
    session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.CLIENT,
            content=f"Меня зовут {NAME}, телефон {PHONE}",
            content_redacted=f"Меня зовут {NAME}, телефон [PHONE]",
        )
    )
    session.add(
        Classification(
            ticket_id=ticket.id, iteration=0, category="faq", risk_level="low", confidence=0.9
        )
    )
    session.add(AuditLog(ticket_id=ticket.id, actor="agent", action="A2", rule_id="R6"))

    if with_operator_reply:
        operator = make_operator(session)
        escalation = Escalation(
            ticket_id=ticket.id,
            reason="low_rag_confidence",
            rule_id="R6",
            status="resolved",
            draft_text=f"{NAME}, доставка займёт 3 дня.",
        )
        session.add(escalation)
        session.flush()
        session.add(
            OperatorAction(
                escalation_id=escalation.id,
                operator_id=operator.id,
                action_type="edit",
                draft_text=f"{NAME}, доставка займёт 3 дня.",
                final_text=f"{NAME}, доставка займёт 3-5 рабочих дней.",
            )
        )
    session.flush()
    return ticket


def ticket_texts(session, ticket_id) -> str:
    """Все текстовые поля тикета, где могли остаться персональные данные."""
    parts = [
        *session.scalars(select(Message.content).where(Message.ticket_id == ticket_id)),
        *session.scalars(select(Message.content_redacted).where(Message.ticket_id == ticket_id)),
        *session.scalars(select(Escalation.draft_text).where(Escalation.ticket_id == ticket_id)),
        *session.scalars(
            select(OperatorAction.final_text)
            .join(Escalation, Escalation.id == OperatorAction.escalation_id)
            .where(Escalation.ticket_id == ticket_id)
        ),
        *session.scalars(
            select(OperatorAction.draft_text)
            .join(Escalation, Escalation.id == OperatorAction.escalation_id)
            .where(Escalation.ticket_id == ticket_id)
        ),
        session.get(Ticket, ticket_id).client_id,
    ]
    return " ".join(str(part) for part in parts if part)


def test_expired_closed_ticket_loses_personal_data(db_session):
    ticket = make_ticket(
        db_session, age_days=91, status=TicketStatus.RESOLVED_BY_OPERATOR, with_operator_reply=True
    )

    result = scrub_expired(db_session, now=NOW, retention=RETENTION)

    assert result.scrubbed == 1
    texts = ticket_texts(db_session, ticket.id)
    assert PHONE not in texts
    assert NAME not in texts
    assert "client-42" not in texts
    assert db_session.get(Ticket, ticket.id).scrubbed_at == NOW


def test_audit_and_metrics_survive_scrub(db_session):
    """Требование NFR4 - «audit_log и агрегаты сохраняются»."""
    ticket = make_ticket(
        db_session, age_days=120, status=TicketStatus.RESOLVED_BY_OPERATOR, with_operator_reply=True
    )
    audit_before = db_session.scalar(
        select(func.count()).select_from(AuditLog).where(AuditLog.ticket_id == ticket.id)
    )

    scrub_expired(db_session, now=NOW, retention=RETENTION)

    refreshed = db_session.get(Ticket, ticket.id)
    assert refreshed.category == "faq"
    assert refreshed.status == TicketStatus.RESOLVED_BY_OPERATOR
    assert db_session.scalar(select(Classification).where(Classification.ticket_id == ticket.id))

    audit = db_session.scalars(select(AuditLog).where(AuditLog.ticket_id == ticket.id)).all()
    assert len(audit) == audit_before + 1
    assert [a.action for a in audit].count("retention_scrub") == 1

    action = db_session.scalar(
        select(OperatorAction)
        .join(Escalation, Escalation.id == OperatorAction.escalation_id)
        .where(Escalation.ticket_id == ticket.id)
    )
    assert action.action_type == "edit"
    assert action.final_text == TOMBSTONE


def test_null_draft_is_preserved_in_database(db_session):
    """NULL в draft_text - «черновика не было» (ADR-008): это метрика, её не затираем."""
    ticket = make_ticket(
        db_session, age_days=100, status=TicketStatus.RESOLVED_BY_OPERATOR, with_operator_reply=True
    )
    action = db_session.scalar(
        select(OperatorAction)
        .join(Escalation, Escalation.id == OperatorAction.escalation_id)
        .where(Escalation.ticket_id == ticket.id)
    )
    action.draft_text = None
    db_session.flush()

    scrub_expired(db_session, now=NOW, retention=RETENTION)

    db_session.expire_all()
    action = db_session.get(OperatorAction, action.id)
    assert action.draft_text is None
    assert action.final_text == TOMBSTONE


def test_fresh_ticket_is_untouched(db_session):
    ticket = make_ticket(db_session, age_days=30, status=TicketStatus.RESOLVED_AUTO)

    result = scrub_expired(db_session, now=NOW, retention=RETENTION)

    assert result.scrubbed == 0
    assert PHONE in ticket_texts(db_session, ticket.id)
    assert db_session.get(Ticket, ticket.id).scrubbed_at is None


def test_open_expired_ticket_is_reported_not_scrubbed(db_session):
    """Оператор ещё работает с тикетом - данные не уничтожаются, но это нарушение NFR4."""
    ticket = make_ticket(db_session, age_days=95, status=TicketStatus.IN_PROGRESS)

    result = scrub_expired(db_session, now=NOW, retention=RETENTION)

    assert result.scrubbed == 0
    assert result.overdue_open == 1
    assert PHONE in ticket_texts(db_session, ticket.id)


def test_scrub_is_idempotent(db_session):
    make_ticket(db_session, age_days=91, status=TicketStatus.RESOLVED_AUTO)

    first = scrub_expired(db_session, now=NOW, retention=RETENTION)
    second = scrub_expired(db_session, now=NOW, retention=RETENTION)

    assert first.scrubbed == 1
    assert second.scrubbed == 0


def test_batch_size_limits_one_run(db_session):
    for _ in range(3):
        make_ticket(db_session, age_days=91, status=TicketStatus.RESOLVED_AUTO)

    assert scrub_expired(db_session, now=NOW, retention=RETENTION, batch_size=2).scrubbed == 2
    assert scrub_expired(db_session, now=NOW, retention=RETENTION, batch_size=2).scrubbed == 1
