"""Клиентский канал: приём (UC1), уточнение, запрос человека (UC9), IDOR (NFR4)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.agent.factory import get_agent_service
from app.agent.graph import TicketGraph
from app.agent.service import AgentService
from app.core.auth import issue_ticket_token, issue_token, webhook_signature
from app.core.config import get_settings
from app.db.base import get_session
from app.db.models import AuditLog, Escalation, Ticket
from app.domain.decision import PRIORITY_CLIENT_REQUESTED, Thresholds
from app.domain.enums import OperatorRole, TicketStatus
from app.main import app
from app.services.classifier import BaselineClassifier
from app.services.embeddings import HashingEmbeddingProvider
from app.services.generation import TemplateResponseGenerator
from app.services.llm import LLMUnavailableError
from app.services.retrieval import Retriever

pytestmark = pytest.mark.integration

WEB_SECRET = get_settings().channel_secrets["web"]


def _agent() -> AgentService:
    graph = TicketGraph(
        BaselineClassifier(),
        Retriever(HashingEmbeddingProvider(), top_k=3),
        TemplateResponseGenerator(),
        Thresholds(rag_confidence=0.15),
    )
    return AgentService(graph)


@pytest.fixture
def client(db_session, indexed_kb):
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[get_agent_service] = _agent
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def post_ticket(client, content, *, external_id=None, channel="web", secret=WEB_SECRET, sign=True):
    body = json.dumps(
        {
            "channel": channel,
            "external_id": external_id or uuid.uuid4().hex,
            "content": content,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if sign:
        headers["X-Signature"] = webhook_signature(body, secret)
    return client.post("/api/v1/tickets", content=body, headers=headers)


def ticket_headers(created: dict) -> dict:
    return {"X-Ticket-Token": created["ticket_token"]}


# --- UC1: приём ---------------------------------------------------------------


def test_faq_ticket_gets_immediate_answer(client):
    response = post_ticket(client, "Подскажите, какие способы оплаты доступны?")

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == TicketStatus.RESOLVED_AUTO
    assert body["reply"]
    assert body["ticket_token"]


def test_high_risk_ticket_is_escalated_without_reply(client, db_session):
    body = post_ticket(client, "Требую вернуть деньги за бракованный товар").json()

    assert body["status"] == TicketStatus.ESCALATED_STANDARD
    assert body["reply"] is None
    ticket_id = uuid.UUID(body["ticket_id"])
    assert db_session.scalar(select(Escalation).where(Escalation.ticket_id == ticket_id))


def test_unsupported_channel_is_rejected_before_anything_else(client, db_session):
    """AC UC1: тикет отклоняется с ошибкой, автоответ не отправляется."""
    before = db_session.scalar(select(func.count()).select_from(Ticket))
    response = post_ticket(client, "Где заказ?", channel="telegram")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_channel"
    assert db_session.scalar(select(func.count()).select_from(Ticket)) == before


@pytest.mark.parametrize(("sign", "secret"), [(False, WEB_SECRET), (True, "wrong-secret")])
def test_unsigned_or_forged_webhook_is_rejected(client, sign, secret):
    response = post_ticket(client, "Где заказ?", sign=sign, secret=secret)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_signature"


def test_malformed_payload_is_400(client):
    body = b'{"channel": "web"}'
    response = client.post(
        "/api/v1/tickets",
        content=body,
        headers={"X-Signature": webhook_signature(body, WEB_SECRET)},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_payload"


def test_webhook_redelivery_returns_same_ticket_without_rerunning_agent(client, db_session):
    first = post_ticket(client, "Где мой заказ?", external_id="wh-42")
    second = post_ticket(client, "Где мой заказ?", external_id="wh-42")

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["ticket_id"] == first.json()["ticket_id"]

    ticket_id = uuid.UUID(first.json()["ticket_id"])
    runs = db_session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.ticket_id == ticket_id, AuditLog.actor == "agent")
    )
    assert runs == 1


def test_llm_outage_escalates_instead_of_failing(client, db_session):
    """NFR6: недоступность LLM - эскалация, а не 500 клиенту."""

    class BrokenAgent:
        def process(self, session, ticket):
            raise LLMUnavailableError("provider down")

    app.dependency_overrides[get_agent_service] = lambda: BrokenAgent()
    body = post_ticket(client, "Где мой заказ?").json()

    assert body["status"] == TicketStatus.ESCALATED_STANDARD
    escalation = db_session.scalar(
        select(Escalation).where(Escalation.ticket_id == uuid.UUID(body["ticket_id"]))
    )
    assert escalation.reason == "llm_unavailable"


# --- NFR4: доступ к тикету ------------------------------------------------------


def test_ticket_is_readable_with_its_token(client):
    created = post_ticket(client, "Подскажите, какие способы оплаты доступны?").json()
    url = f"/api/v1/tickets/{created['ticket_id']}"
    response = client.get(url, headers=ticket_headers(created))

    assert response.status_code == 200
    senders = [m["sender"] for m in response.json()["messages"]]
    assert senders == ["client", "agent"]
    # Внутренности решения клиенту не показываются.
    assert "category" not in response.json()


def test_foreign_ticket_cannot_be_read_with_own_token(client):
    """IDOR: токен своего тикета не открывает чужой - и не выдаёт, что он существует."""
    mine = post_ticket(client, "Где мой заказ?").json()
    theirs = post_ticket(client, "Где мой заказ?").json()

    response = client.get(f"/api/v1/tickets/{theirs['ticket_id']}", headers=ticket_headers(mine))
    assert response.status_code == 404


def test_operator_token_is_not_a_ticket_token(client):
    """Токены подписаны одним секретом - тип должен проверяться."""
    created = post_ticket(client, "Где мой заказ?").json()
    operator_token = issue_token(uuid.UUID(created["ticket_id"]), OperatorRole.ADMIN)

    response = client.get(
        f"/api/v1/tickets/{created['ticket_id']}", headers={"X-Ticket-Token": operator_token}
    )
    assert response.status_code == 404


def test_ticket_token_does_not_open_operator_api(client):
    created = post_ticket(client, "Где мой заказ?").json()
    response = client.get(
        "/api/v1/escalations", headers={"Authorization": f"Bearer {created['ticket_token']}"}
    )
    assert response.status_code == 401


def test_missing_ticket_token_is_401(client):
    created = post_ticket(client, "Где мой заказ?").json()
    assert client.get(f"/api/v1/tickets/{created['ticket_id']}").status_code == 401


# --- уточнение ----------------------------------------------------------------


def test_answer_to_clarification_reruns_agent_and_hits_limit(client, db_session):
    created = post_ticket(client, "Приложение вылетает при запуске").json()
    assert created["status"] == TicketStatus.AWAITING_CLARIFICATION

    response = client.post(
        f"/api/v1/tickets/{created['ticket_id']}/messages",
        json={"content": "Android 14, вылетает сразу на заставке"},
        headers=ticket_headers(created),
    )

    assert response.status_code == 200
    assert response.json()["status"] == TicketStatus.ESCALATED_STANDARD
    ticket = db_session.get(Ticket, uuid.UUID(created["ticket_id"]))
    assert ticket.last_client_reply_at is not None


def test_message_to_escalated_ticket_does_not_run_agent(client, db_session):
    created = post_ticket(client, "Требую вернуть деньги за бракованный товар").json()
    ticket_id = uuid.UUID(created["ticket_id"])
    runs_before = db_session.scalar(
        select(func.count()).select_from(AuditLog).where(AuditLog.ticket_id == ticket_id)
    )

    response = client.post(
        f"/api/v1/tickets/{created['ticket_id']}/messages",
        json={"content": "Прилагаю фото"},
        headers=ticket_headers(created),
    )

    assert response.status_code == 200
    assert response.json()["reply"] is None
    runs_after = db_session.scalar(
        select(func.count()).select_from(AuditLog).where(AuditLog.ticket_id == ticket_id)
    )
    assert runs_after == runs_before


def test_message_to_closed_ticket_is_rejected(client):
    created = post_ticket(client, "Подскажите, какие способы оплаты доступны?").json()
    response = client.post(
        f"/api/v1/tickets/{created['ticket_id']}/messages",
        json={"content": "Спасибо"},
        headers=ticket_headers(created),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ticket_closed"


# --- UC9: запрос человека ------------------------------------------------------


def test_client_can_escalate_after_auto_answer(client, db_session):
    created = post_ticket(client, "Подскажите, какие способы оплаты доступны?").json()
    assert created["status"] == TicketStatus.RESOLVED_AUTO

    response = client.post(
        f"/api/v1/tickets/{created['ticket_id']}/escalate", headers=ticket_headers(created)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == TicketStatus.ESCALATED_PRIORITY
    assert body["priority"] == PRIORITY_CLIENT_REQUESTED

    escalation = db_session.get(Escalation, uuid.UUID(body["escalation_id"]))
    assert escalation.rule_id == "R1"
    assert escalation.reason == "client_requested"
    assert db_session.get(Ticket, uuid.UUID(created["ticket_id"])).resolved_at is None


def test_escalation_window_after_auto_answer_expires(client, db_session):
    created = post_ticket(client, "Подскажите, какие способы оплаты доступны?").json()
    ticket = db_session.get(Ticket, uuid.UUID(created["ticket_id"]))
    ticket.resolved_at = datetime.now(UTC) - timedelta(hours=25)
    db_session.flush()

    response = client.post(
        f"/api/v1/tickets/{created['ticket_id']}/escalate", headers=ticket_headers(created)
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "escalation_window_expired"


def test_repeated_escalate_raises_priority_of_existing_escalation(client, db_session):
    """Тикет уже ждёт человека: вторую эскалацию не создаём, поднимаем приоритет."""
    created = post_ticket(client, "Требую вернуть деньги за бракованный товар").json()
    ticket_id = uuid.UUID(created["ticket_id"])

    response = client.post(
        f"/api/v1/tickets/{created['ticket_id']}/escalate", headers=ticket_headers(created)
    )

    escalations = db_session.scalars(
        select(Escalation).where(Escalation.ticket_id == ticket_id)
    ).all()
    assert len(escalations) == 1
    assert escalations[0].priority == PRIORITY_CLIENT_REQUESTED
    assert response.json()["status"] == TicketStatus.ESCALATED_PRIORITY


def test_ticket_token_for_nonexistent_ticket_is_404(client):
    ghost = uuid.uuid4()
    response = client.get(
        f"/api/v1/tickets/{ghost}", headers={"X-Ticket-Token": issue_ticket_token(ghost)}
    )
    assert response.status_code == 404
