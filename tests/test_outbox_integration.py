"""Интеграционные тесты outbox-поллера и consumer'а (ADR-004, ADR-007) без брокера."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.db.models import AuditLog, ConsumedEvent, OutboxEvent, Ticket
from app.domain.enums import TicketStatus
from app.escalations.consumer import (
    ConsumeResult,
    PoisonMessageError,
    handle_escalation_created,
)
from app.messaging.outbox import MAX_ATTEMPTS, publish_pending
from app.messaging.publisher import PublishError
from tests.escalation_fixtures import make_escalated_ticket

pytestmark = pytest.mark.integration


class RecordingPublisher:
    def __init__(self, fail_after: int | None = None) -> None:
        self.sent: list[dict] = []
        self._fail_after = fail_after

    def publish(self, routing_key, body, *, message_id, priority=0):
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            raise PublishError("broker down")
        self.sent.append(
            {
                "routing_key": routing_key,
                "body": json.loads(body),
                "message_id": message_id,
                "priority": priority,
            }
        )


def _event(session, escalation) -> OutboxEvent:
    return session.scalar(
        select(OutboxEvent).where(OutboxEvent.payload["escalation_id"].astext == str(escalation.id))
    )


# --- поллер -----------------------------------------------------------------


def test_pending_events_are_published_and_marked(db_session):
    _, escalation = make_escalated_ticket(db_session, priority=10)
    publisher = RecordingPublisher()

    result = publish_pending(db_session, publisher)

    assert result.published >= 1
    sent = [m for m in publisher.sent if m["body"]["escalation_id"] == str(escalation.id)]
    assert len(sent) == 1
    assert sent[0]["message_id"] == f"escalation.created:{escalation.id}"
    assert sent[0]["priority"] == 10
    assert sent[0]["routing_key"] == "escalation.created"

    event = _event(db_session, escalation)
    assert event.published is True
    assert event.published_at is not None


def test_published_events_are_not_sent_again(db_session):
    make_escalated_ticket(db_session)
    publish_pending(db_session, RecordingPublisher())

    second = RecordingPublisher()
    assert publish_pending(db_session, second).published == 0
    assert second.sent == []


def test_publish_failure_keeps_event_and_records_error(db_session):
    _, escalation = make_escalated_ticket(db_session)
    result = publish_pending(db_session, RecordingPublisher(fail_after=0))

    event = _event(db_session, escalation)
    assert result.failed == 1
    assert event.published is False
    assert event.attempts == 1
    assert "broker down" in event.last_error


def test_batch_stops_at_first_failure(db_session):
    """Сбой брокера обычно общий - остаток пачки не долбит его зря."""
    for _ in range(3):
        make_escalated_ticket(db_session)
    publisher = RecordingPublisher(fail_after=1)

    result = publish_pending(db_session, publisher)

    assert result.published == 1
    assert result.failed == 1
    unpublished = db_session.scalars(
        select(OutboxEvent).where(OutboxEvent.published.is_(False))
    ).all()
    assert len(unpublished) == 2


def test_exhausted_event_does_not_block_the_queue(db_session):
    _, poisoned = make_escalated_ticket(db_session)
    _event(db_session, poisoned).attempts = MAX_ATTEMPTS
    _, healthy = make_escalated_ticket(db_session)
    db_session.flush()

    publisher = RecordingPublisher()
    result = publish_pending(db_session, publisher)

    sent_ids = {m["body"]["escalation_id"] for m in publisher.sent}
    assert str(healthy.id) in sent_ids
    assert str(poisoned.id) not in sent_ids
    assert result.stuck == 1


def test_oldest_pending_age_is_reported(db_session):
    make_escalated_ticket(db_session)
    later = datetime.now(UTC) + timedelta(minutes=5)
    result = publish_pending(db_session, RecordingPublisher(fail_after=0), now=later)
    assert result.oldest_pending_age_seconds >= 290


# --- consumer ---------------------------------------------------------------


def _deliver(session, escalation):
    event = _event(session, escalation)
    return handle_escalation_created(
        session, idempotency_key=event.idempotency_key, payload=event.payload
    )


def test_consumer_moves_ticket_to_operator_queue(db_session):
    ticket, escalation = make_escalated_ticket(db_session)

    assert _deliver(db_session, escalation) is ConsumeResult.PROCESSED
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.PENDING_OPERATOR

    audit = db_session.scalar(
        select(AuditLog).where(
            AuditLog.ticket_id == ticket.id, AuditLog.action == "queued_for_operator"
        )
    )
    assert audit.payload["status_before"] == TicketStatus.ESCALATED_STANDARD


def test_redelivery_is_a_no_op(db_session):
    """At-least-once у поллера закрывается inbox'ом consumer'а (ADR-007)."""
    ticket, escalation = make_escalated_ticket(db_session)

    assert _deliver(db_session, escalation) is ConsumeResult.PROCESSED
    assert _deliver(db_session, escalation) is ConsumeResult.DUPLICATE

    audits = db_session.scalars(
        select(AuditLog).where(
            AuditLog.ticket_id == ticket.id, AuditLog.action == "queued_for_operator"
        )
    ).all()
    assert len(audits) == 1
    assert db_session.scalar(
        select(ConsumedEvent).where(
            ConsumedEvent.idempotency_key == f"escalation.created:{escalation.id}"
        )
    )


def test_consumer_does_not_move_ticket_backwards(db_session):
    """Оператор мог взять тикет по REST раньше, чем пришло событие."""
    ticket, escalation = make_escalated_ticket(db_session, status=TicketStatus.IN_PROGRESS)
    _deliver(db_session, escalation)
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.IN_PROGRESS


@pytest.mark.parametrize(
    ("key", "payload"),
    [
        (None, {"escalation_id": "00000000-0000-0000-0000-000000000000"}),
        ("k1", {}),
        ("k2", {"escalation_id": "not-a-uuid"}),
        ("k3", {"escalation_id": "00000000-0000-0000-0000-000000000000"}),
    ],
)
def test_poison_messages_are_rejected(db_session, key, payload):
    with pytest.raises(PoisonMessageError):
        handle_escalation_created(db_session, idempotency_key=key, payload=payload)
