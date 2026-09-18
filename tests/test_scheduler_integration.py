"""Шедулер: таймаут уточнения (NFR9) и истечение claim'ов (FR10)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.db.models import Escalation, Message, OutboxEvent, Ticket
from app.domain.enums import MessageSender, TicketStatus
from app.escalations import operations as ops
from app.escalations.timeouts import escalate_clarification_timeouts
from tests.escalation_fixtures import make_escalated_ticket, make_operator

pytestmark = pytest.mark.integration

TIMEOUT = timedelta(minutes=30)


def _awaiting_ticket(session, asked_minutes_ago: int) -> Ticket:
    ticket = Ticket(
        channel="web",
        external_id=f"clar-{asked_minutes_ago}-{id(session)}",
        status=TicketStatus.AWAITING_CLARIFICATION.value,
        category="tech_issue",
        clarification_count=1,
    )
    session.add(ticket)
    session.flush()
    session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.AGENT,
            content="Уточните, пожалуйста, детали",
            created_at=datetime.now(UTC) - timedelta(minutes=asked_minutes_ago),
        )
    )
    session.flush()
    return ticket


def test_silent_client_is_escalated_after_timeout(db_session):
    ticket = _awaiting_ticket(db_session, asked_minutes_ago=31)

    escalated = escalate_clarification_timeouts(
        db_session, now=datetime.now(UTC), timeout=TIMEOUT
    )

    assert escalated == 1
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.ESCALATED_STANDARD
    escalation = db_session.scalar(select(Escalation).where(Escalation.ticket_id == ticket.id))
    assert escalation.reason == "clarification_timeout"
    assert escalation.rule_id == "R7b"


def test_timeout_escalation_writes_outbox_event_too(db_session):
    """Эскалация из шедулера идёт тем же транзакционным путём, что и из агента."""
    ticket = _awaiting_ticket(db_session, asked_minutes_ago=45)
    escalate_clarification_timeouts(db_session, now=datetime.now(UTC), timeout=TIMEOUT)

    event = db_session.scalar(select(OutboxEvent).where(OutboxEvent.ticket_id == ticket.id))
    assert event is not None
    assert event.payload["reason"] == "clarification_timeout"


def test_client_still_has_time(db_session):
    ticket = _awaiting_ticket(db_session, asked_minutes_ago=10)
    assert escalate_clarification_timeouts(
        db_session, now=datetime.now(UTC), timeout=TIMEOUT
    ) == 0
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.AWAITING_CLARIFICATION


def test_timeout_is_idempotent(db_session):
    _awaiting_ticket(db_session, asked_minutes_ago=31)
    now = datetime.now(UTC)
    assert escalate_clarification_timeouts(db_session, now=now, timeout=TIMEOUT) == 1
    assert escalate_clarification_timeouts(db_session, now=now, timeout=TIMEOUT) == 0


def test_expired_claims_return_to_queue(db_session):
    anna = make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session)
    now = datetime.now(UTC)
    ops.claim(db_session, escalation.id, anna.id, now=now, ttl=timedelta(minutes=15))

    assert ops.release_expired(db_session, now=now + timedelta(minutes=5)) == 0
    assert ops.release_expired(db_session, now=now + timedelta(minutes=16)) == 1

    refreshed = db_session.get(Escalation, escalation.id)
    assert refreshed.status == "pending"
    assert refreshed.locked_by is None
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.PENDING_OPERATOR
