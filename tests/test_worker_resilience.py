"""Воркеры пережидают временные сбои Postgres и RabbitMQ, а не падают."""

from __future__ import annotations

import pika.exceptions
import pytest
from sqlalchemy.exc import OperationalError

from app.messaging.outbox import PollResult
from app.workers import escalation_consumer, outbox_poller
from app.workers.backoff import Backoff


class StopLoop(Exception):
    """Прерывает бесконечный цикл воркера в тесте."""


def db_down() -> OperationalError:
    return OperationalError("SELECT 1", {}, ConnectionRefusedError("connection refused"))


class SleepRecorder:
    def __init__(self, stop_after: int) -> None:
        self.calls: list[float] = []
        self._stop_after = stop_after

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if len(self.calls) >= self._stop_after:
            raise StopLoop


# --- Backoff -----------------------------------------------------------------


def test_backoff_grows_exponentially_up_to_cap():
    backoff = Backoff(initial=1, maximum=10, jitter=False)
    assert [backoff.next_delay() for _ in range(6)] == [1, 2, 4, 8, 10, 10]
    assert backoff.attempt == 6


def test_backoff_reset_starts_over():
    backoff = Backoff(initial=1, jitter=False)
    backoff.next_delay()
    backoff.next_delay()
    backoff.reset()
    assert backoff.attempt == 0
    assert backoff.next_delay() == 1


def test_backoff_jitter_stays_within_half_to_full_delay():
    for _ in range(200):
        backoff = Backoff(initial=4, jitter=True)
        assert 2 <= backoff.next_delay() <= 4


# --- outbox poller ---------------------------------------------------------------


class FakePublisher:
    def __init__(self) -> None:
        self.keepalives = 0

    def keepalive(self) -> None:
        self.keepalives += 1


class NullSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_poller_survives_database_outage(monkeypatch):
    outcomes = [
        db_down(),
        db_down(),
        PollResult(published=0, failed=0, stuck=0, oldest_pending_age_seconds=None),
    ]

    def fake_publish_pending(session, publisher, *, batch_size):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleep = SleepRecorder(stop_after=3)
    monkeypatch.setattr(outbox_poller, "publish_pending", fake_publish_pending)
    monkeypatch.setattr(outbox_poller.time, "sleep", sleep)
    backoff = Backoff(initial=1, jitter=False)
    publisher = FakePublisher()

    with pytest.raises(StopLoop):
        outbox_poller.run(NullSession, publisher, interval=0.5, batch_size=10, backoff=backoff)

    # Две паузы на недоступную базу, затем обычный простой с keepalive.
    assert sleep.calls == [1, 2, 0.5]
    assert backoff.attempt == 0
    assert publisher.keepalives == 1


# --- escalation consumer ---------------------------------------------------------


def test_consumer_reconnects_after_connection_loss(monkeypatch):
    failures = [
        pika.exceptions.AMQPConnectionError("broker is down"),
        pika.exceptions.StreamLostError("connection reset"),
        pika.exceptions.ChannelWrongStateError("channel is closed"),
    ]
    connected_at_attempt: list[int] = []
    backoff = Backoff(initial=1, jitter=False)

    def fake_consume(session_factory, url, *, on_connected):
        if failures:
            raise failures.pop(0)
        connected_at_attempt.append(backoff.attempt)
        on_connected()  # соединение поднялось; дальше - штатная остановка

    sleep = SleepRecorder(stop_after=100)
    monkeypatch.setattr(escalation_consumer, "consume", fake_consume)
    monkeypatch.setattr(escalation_consumer.time, "sleep", sleep)

    escalation_consumer.run(NullSession, "amqp://", backoff=backoff)

    assert sleep.calls == [1, 2, 4]
    assert connected_at_attempt == [3]
    assert backoff.attempt == 0


def test_consumer_fails_loudly_on_channel_closed_by_broker(monkeypatch):
    """Канал закрыт брокером (например, расхождение топологии) - не временный сбой."""

    def fake_consume(session_factory, url, *, on_connected):
        raise pika.exceptions.ChannelClosedByBroker(406, "PRECONDITION_FAILED")

    monkeypatch.setattr(escalation_consumer, "consume", fake_consume)
    monkeypatch.setattr(escalation_consumer.time, "sleep", SleepRecorder(stop_after=1))

    with pytest.raises(pika.exceptions.ChannelClosedByBroker):
        escalation_consumer.run(NullSession, "amqp://", backoff=Backoff(jitter=False))


class DeadConnection:
    """Соединение, которое брокер уже закрыл."""

    is_open = False

    def close(self) -> None:
        raise pika.exceptions.ConnectionWrongStateError("already closed")


def test_consume_does_not_mask_original_error_with_close(monkeypatch):
    """Регрессия: close() на мёртвом соединении подменял исходную ошибку своей."""
    monkeypatch.setattr(
        escalation_consumer.pika, "BlockingConnection", lambda params: DeadConnection()
    )

    def lost(*args):
        raise pika.exceptions.StreamLostError("connection reset")

    monkeypatch.setattr(escalation_consumer, "_consume_on", lost)

    with pytest.raises(pika.exceptions.StreamLostError):
        escalation_consumer.consume(NullSession, "amqp://guest:guest@localhost/")
