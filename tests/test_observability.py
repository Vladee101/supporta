"""Наблюдаемость: сквозной trace_id, модель ошибок, структурные логи, метрики SLI."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import metrics
from app.core import tracing
from app.core.log_setup import JsonFormatter, RedactingTextFormatter, TraceFilter
from app.domain.decision import Decision
from app.domain.enums import Action, EscalationReason
from app.main import app

PHONE = "+7 999 123-45-67"
EMAIL = "ivan.petrov@example.com"


# --- trace_id и модель ошибок ------------------------------------------------


def test_trace_id_is_generated_and_returned_in_header():
    with TestClient(app) as client:
        response = client.get("/health")
    assert len(response.headers["x-trace-id"]) == 32


def test_valid_incoming_trace_id_is_kept():
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Trace-Id": "0123456789abcdef"})
    assert response.headers["x-trace-id"] == "0123456789abcdef"


def test_arbitrary_header_value_is_not_trusted():
    """Иначе заголовок стал бы инъекцией в логи."""
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Trace-Id": "evil value; level=ERROR"})
    assert response.headers["x-trace-id"] != "evil value; level=ERROR"
    assert len(response.headers["x-trace-id"]) == 32


def test_routing_errors_use_the_single_error_model():
    """404 роутинга FastAPI по умолчанию отдаёт как {"detail": ...} - без кода и trace_id."""
    with TestClient(app) as client:
        response = client.get("/no-such-route", headers={"X-Trace-Id": "abcdef0123456789"})

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_404"
    assert error["trace_id"] == "abcdef0123456789" == response.headers["x-trace-id"]


def test_trace_id_outside_request_is_generated():
    assert len(tracing.ensure_trace_id()) == 32


# --- логи ---------------------------------------------------------------------


def _record(message, *args, exc_info=None) -> logging.LogRecord:
    record = logging.LogRecord("support.test", logging.INFO, __file__, 1, message, args, exc_info)
    TraceFilter().filter(record)
    return record


def test_json_log_line_is_structured_and_redacted():
    tracing.set_trace_id("feedface00000000")
    line = JsonFormatter().format(_record("клиент %s оставил телефон %s", EMAIL, PHONE))
    entry = json.loads(line)

    assert entry["trace_id"] == "feedface00000000"
    assert entry["level"] == "INFO"
    assert PHONE not in line and EMAIL not in line


def test_exception_text_is_redacted_too():
    """Текст обращения легко оказывается в трейсбеке - редакция сообщения его не поймает."""
    try:
        raise ValueError(f"не удалось обработать: {PHONE}")
    except ValueError:
        line = JsonFormatter().format(_record("сбой обработки", exc_info=sys.exc_info()))

    assert "ValueError" in json.loads(line)["exception"]
    assert PHONE not in line


def test_text_format_is_redacted_as_well():
    formatter = RedactingTextFormatter("%(message)s [%(trace_id)s]")
    assert PHONE not in formatter.format(_record("телефон %s", PHONE))


# --- метрики ------------------------------------------------------------------


def _value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _outcome(action: Action, rule_id: str, reason=None, usage=None):
    return SimpleNamespace(
        decision=Decision(action=action, rule_id=rule_id, reason=reason), llm_usage=usage
    )


def test_ticket_metrics_count_rules_reasons_and_llm_cost():
    before_r4 = _value("support_tickets_processed_total", action="A1", rule_id="R4")
    before_rag = _value("support_escalations_total", reason="low_rag_confidence")
    before_cost = _value("support_llm_cost_rub_total")

    metrics.observe_ticket(
        _outcome(
            Action.AUTO_ANSWER,
            "R4",
            usage={
                "calls": 2,
                "prompt_tokens": 700,
                "completion_tokens": 60,
                "cost": 0.1,
                "calls_without_cost": 0,
                "by_model": {"m": 2},
            },
        ),
        seconds=1.5,
    )
    metrics.observe_ticket(
        _outcome(Action.ESCALATE, "R6", reason=EscalationReason.LOW_RAG_CONFIDENCE), seconds=10
    )

    assert _value("support_tickets_processed_total", action="A1", rule_id="R4") == before_r4 + 1
    assert _value("support_escalations_total", reason="low_rag_confidence") == before_rag + 1
    assert _value("support_llm_cost_rub_total") == before_cost + 0.1


def test_llm_outage_path_is_counted():
    """Эскалация по NFR6 идёт в обход графа - без отдельного учёта SLI её бы не видел."""
    before = _value("support_escalations_total", reason="llm_unavailable")
    metrics.observe_llm_unavailable()
    assert _value("support_escalations_total", reason="llm_unavailable") == before + 1
    assert _value("support_tickets_processed_total", action="A2", rule_id="NFR6") >= 1


def test_operator_actions_are_counted_by_type():
    before = _value("support_operator_actions_total", action_type="edit")
    metrics.observe_operator_action("edit")
    assert _value("support_operator_actions_total", action_type="edit") == before + 1


def test_metrics_endpoint_serves_prometheus_text():
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "support_tickets_processed_total" in response.text


def test_state_collector_reports_unreachable_sources_instead_of_failing():
    """База и брокер недоступны - это значение метрики (up = 0), а не 500 на /metrics."""

    def broken_session():
        raise ConnectionError("database is down")

    collector = metrics.StateCollector(
        broken_session, "amqp://guest:guest@127.0.0.1:1/%2F?connection_attempts=1"
    )
    families = {family.name: family for family in collector.collect()}

    assert families["support_database_up"].samples[0].value == 0
    assert families["support_rabbitmq_up"].samples[0].value == 0


@pytest.mark.integration
def test_state_collector_reads_outbox_age_from_database(db_session):
    from datetime import UTC, datetime, timedelta

    from app.db.models import OutboxEvent, Ticket

    ticket = Ticket(channel="web", external_id="metrics-1", status="escalated_standard")
    db_session.add(ticket)
    db_session.flush()
    db_session.add(
        OutboxEvent(
            ticket_id=ticket.id,
            event_type="escalation.created",
            idempotency_key="metrics-1",
            payload={},
            created_at=datetime.now(UTC) - timedelta(minutes=5),
        )
    )
    db_session.flush()

    collector = metrics.StateCollector(lambda: nullcontext(db_session), "amqp://127.0.0.1:1/")
    families = {family.name: family for family in collector._outbox()}

    assert families["support_outbox_pending"].samples[0].value >= 1
    assert families["support_outbox_oldest_pending_seconds"].samples[0].value >= 290
