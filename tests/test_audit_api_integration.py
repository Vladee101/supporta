"""Audit trail (UC8, NFR3)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agent.graph import TicketGraph
from app.agent.service import AgentService
from app.core.auth import issue_token
from app.db.base import get_session
from app.db.models import Message, Ticket
from app.domain.decision import Thresholds
from app.domain.enums import MessageSender, OperatorRole
from app.main import app
from app.services.classifier import BaselineClassifier
from app.services.embeddings import HashingEmbeddingProvider
from app.services.generation import TemplateResponseGenerator
from app.services.retrieval import Retriever
from tests.escalation_fixtures import make_operator

pytestmark = pytest.mark.integration


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_session] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_audit_shows_both_clarification_iterations(client, db_session, indexed_kb):
    """AC UC8: обе итерации видны отдельно, с пометкой о сработавшем лимите."""
    agent = AgentService(
        TicketGraph(
            BaselineClassifier(),
            Retriever(HashingEmbeddingProvider(), top_k=3),
            TemplateResponseGenerator(),
            Thresholds(rag_confidence=0.15),
        )
    )
    ticket = Ticket(channel="web", external_id=uuid.uuid4().hex, status="new")
    db_session.add(ticket)
    db_session.flush()
    db_session.add(
        Message(ticket_id=ticket.id, sender=MessageSender.CLIENT, content="Приложение вылетает")
    )
    db_session.flush()
    agent.process(db_session, ticket)
    db_session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.CLIENT,
            iteration=1,
            content="Вылетает при запуске приложения",
        )
    )
    db_session.flush()
    agent.process(db_session, ticket)

    operator = make_operator(db_session)
    headers = {"Authorization": f"Bearer {issue_token(operator.id, OperatorRole.OPERATOR)}"}
    body = client.get(f"/api/v1/tickets/{ticket.id}/audit", headers=headers).json()

    assert [c["iteration"] for c in body["classifications"]] == [0, 1]
    assert [e["rule_id"] for e in body["events"] if e["actor"] == "agent"] == ["R7a", "R7b"]
    assert body["clarification_limit_reached"] is True
    agent_events = [e for e in body["events"] if e["actor"] == "agent"]
    assert all(e["trace_id"] for e in agent_events)
    assert all(e["rag_confidence"] is not None for e in agent_events)


def test_audit_requires_operator_token(client, db_session):
    ticket_id = db_session.scalar(select(Ticket.id).limit(1)) or uuid.uuid4()
    assert client.get(f"/api/v1/tickets/{ticket_id}/audit").status_code == 401


def test_audit_of_unknown_ticket_is_404(client, db_session):
    operator = make_operator(db_session)
    headers = {"Authorization": f"Bearer {issue_token(operator.id, OperatorRole.OPERATOR)}"}
    assert client.get(f"/api/v1/tickets/{uuid.uuid4()}/audit", headers=headers).status_code == 404
