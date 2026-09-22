"""API Operator Console: очередь, контекст, claim, release, resolve (UC6, FR10, NFR7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.auth import issue_token
from app.db.base import get_session
from app.db.models import (
    AuditLog,
    Escalation,
    KbDocument,
    KbDocumentVersion,
    Message,
    OperatorAction,
    RagRetrieval,
    Ticket,
)
from app.domain.enums import EscalationReason, OperatorRole, TicketStatus
from app.escalations import operations as ops
from app.main import app
from tests.escalation_fixtures import make_escalated_ticket, make_operator

pytestmark = pytest.mark.integration


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_session] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def auth(operator) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(operator.id, OperatorRole(operator.role))}"}


# --- доступ -----------------------------------------------------------------


def test_queue_requires_token(client):
    response = client.get("/api/v1/escalations")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.json()["error"]["trace_id"]


# --- очередь ----------------------------------------------------------------


def test_queue_orders_by_priority_then_age(client, db_session):
    operator = make_operator(db_session)
    _, standard_old = make_escalated_ticket(db_session, priority=0)
    _, priority = make_escalated_ticket(db_session, priority=10)
    _, standard_new = make_escalated_ticket(db_session, priority=0)
    # created_at внутри одной транзакции совпадает (now()) - разводим явно.
    base = datetime.now(UTC)
    standard_old.created_at = base - timedelta(minutes=10)
    standard_new.created_at = base - timedelta(minutes=1)
    db_session.flush()

    page = client.get("/api/v1/escalations", headers=auth(operator)).json()
    ids = [item["id"] for item in page["items"]]
    assert ids == [str(priority.id), str(standard_old.id), str(standard_new.id)]


def test_queue_keyset_pagination_covers_everything_once(client, db_session):
    operator = make_operator(db_session)
    created = []
    base = datetime.now(UTC)
    for minute in range(5):
        _, escalation = make_escalated_ticket(db_session)
        escalation.created_at = base - timedelta(minutes=10 - minute)
        created.append(str(escalation.id))
    db_session.flush()

    seen, cursor = [], None
    while True:
        params = {"limit": 2} | ({"cursor": cursor} if cursor else {})
        page = client.get("/api/v1/escalations", params=params, headers=auth(operator)).json()
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if not cursor:
            break

    assert seen == created


def test_context_package_contains_what_operator_needs(client, db_session):
    operator = make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session)

    body = client.get(f"/api/v1/escalations/{escalation.id}", headers=auth(operator)).json()

    assert body["escalation"]["reason"] == EscalationReason.LOW_RAG_CONFIDENCE
    assert body["escalation"]["draft_text"]
    assert body["ticket"]["id"] == str(ticket.id)
    assert body["messages"][0]["content"] == "Вопрос клиента"


def test_context_documents_are_named_not_just_ranked(client, db_session):
    """Оператор видит, какой документ нашёл агент, не раскрывая каждый снапшот."""
    operator = make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session)
    document = KbDocument(slug="delivery-terms", title="Сроки доставки")
    db_session.add(document)
    db_session.flush()
    version = KbDocumentVersion(document_id=document.id, version=2, content="Текст версии 2")
    db_session.add(version)
    db_session.flush()
    db_session.add(
        RagRetrieval(
            ticket_id=ticket.id,
            document_version_id=version.id,
            rank=1,
            relevance_score=0.81,
            chunk_snapshot="То, что видел агент",
        )
    )
    db_session.flush()

    body = client.get(f"/api/v1/escalations/{escalation.id}", headers=auth(operator)).json()

    [found] = body["documents"]
    assert found["title"] == "Сроки доставки"
    assert (found["slug"], found["version"]) == ("delivery-terms", 2)
    assert found["snapshot"] == "То, что видел агент"


def test_unknown_escalation_is_404(client, db_session):
    operator = make_operator(db_session)
    response = client.get(
        "/api/v1/escalations/00000000-0000-0000-0000-000000000000", headers=auth(operator)
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --- claim ------------------------------------------------------------------


def test_claim_locks_escalation_for_one_operator(client, db_session):
    anna, boris = make_operator(db_session), make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session)

    first = client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(anna))
    second = client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(boris))

    assert first.status_code == 200
    assert first.json()["locked_by"] == str(anna.id)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "already_locked"
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.IN_PROGRESS


def test_claimed_escalation_leaves_the_queue(client, db_session):
    anna = make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)
    client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(anna))

    page = client.get("/api/v1/escalations", headers=auth(anna)).json()
    assert str(escalation.id) not in [item["id"] for item in page["items"]]


def test_reclaim_by_owner_extends_the_lock(client, db_session):
    anna = make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)

    first = client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(anna)).json()
    again = client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(anna))

    assert again.status_code == 200
    assert again.json()["locked_until"] >= first["locked_until"]


def test_expired_claim_can_be_taken_over(db_session):
    """Ленивое истечение: истёкший claim свободен сразу, без шедулера."""
    anna, boris = make_operator(db_session), make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)
    now = datetime.now(UTC)

    ops.claim(db_session, escalation.id, anna.id, now=now, ttl=timedelta(minutes=15))
    later = now + timedelta(minutes=16)
    taken = ops.claim(db_session, escalation.id, boris.id, now=later, ttl=timedelta(minutes=15))

    assert taken.locked_by == boris.id


def test_release_returns_escalation_to_queue(client, db_session):
    anna, boris = make_operator(db_session), make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session)
    client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(anna))

    foreign = client.delete(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(boris))
    own = client.delete(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(anna))

    assert foreign.status_code == 409
    assert own.status_code == 200
    assert own.json()["locked_by"] is None
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.PENDING_OPERATOR


# --- resolve ----------------------------------------------------------------


def _claim(client, escalation, operator):
    response = client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(operator))
    assert response.status_code == 200


def test_resolve_requires_claim(client, db_session):
    anna = make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)

    response = client.post(
        f"/api/v1/escalations/{escalation.id}/resolve",
        json={"action": "confirm"},
        headers=auth(anna),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_claimed"


def test_confirm_sends_draft_as_is(client, db_session):
    anna = make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session, draft="Черновик ответа агента.")
    _claim(client, escalation, anna)

    response = client.post(
        f"/api/v1/escalations/{escalation.id}/resolve",
        json={"action": "confirm"},
        headers=auth(anna),
    )

    assert response.status_code == 200
    assert response.json()["final_text"] == "Черновик ответа агента."
    assert db_session.get(Ticket, ticket.id).status == TicketStatus.RESOLVED_BY_OPERATOR
    assert db_session.get(Escalation, escalation.id).status == "resolved"


def test_edit_stores_draft_and_final_for_diff(client, db_session):
    """NFR7: правка оператора сохраняется парой draft/final."""
    anna = make_operator(db_session)
    ticket, escalation = make_escalated_ticket(db_session, draft="Доставка 3 дня.")
    _claim(client, escalation, anna)

    client.post(
        f"/api/v1/escalations/{escalation.id}/resolve",
        json={"action": "edit", "final_text": "Доставка займёт 3-5 рабочих дней."},
        headers=auth(anna),
    )

    action = db_session.scalar(
        select(OperatorAction).where(OperatorAction.escalation_id == escalation.id)
    )
    assert action.draft_text == "Доставка 3 дня."
    assert action.final_text == "Доставка займёт 3-5 рабочих дней."
    assert action.operator_id == anna.id

    reply = db_session.scalar(
        select(Message).where(Message.ticket_id == ticket.id, Message.sender == "operator")
    )
    assert reply.content == action.final_text

    audit = db_session.scalar(
        select(AuditLog).where(AuditLog.ticket_id == ticket.id, AuditLog.action == "resolve:edit")
    )
    assert 0.0 < audit.payload["draft_similarity"] < 1.0


def test_confirm_is_impossible_without_draft(client, db_session):
    """ADR-008: у жалоб и возвратов черновика нет, подтверждать нечего."""
    anna = make_operator(db_session)
    _, escalation = make_escalated_ticket(
        db_session, draft=None, reason=EscalationReason.HIGH_RISK_CATEGORY, category="refund"
    )
    _claim(client, escalation, anna)

    response = client.post(
        f"/api/v1/escalations/{escalation.id}/resolve",
        json={"action": "confirm"},
        headers=auth(anna),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_resolution"


@pytest.mark.parametrize("action", ["edit", "reject"])
def test_edit_and_reject_require_text(client, db_session, action):
    anna = make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)
    _claim(client, escalation, anna)

    response = client.post(
        f"/api/v1/escalations/{escalation.id}/resolve",
        json={"action": action, "final_text": "   "},
        headers=auth(anna),
    )
    assert response.status_code == 422


def test_resolved_escalation_cannot_be_claimed_again(client, db_session):
    anna, boris = make_operator(db_session), make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)
    _claim(client, escalation, anna)
    client.post(
        f"/api/v1/escalations/{escalation.id}/resolve",
        json={"action": "confirm"},
        headers=auth(anna),
    )

    response = client.post(f"/api/v1/escalations/{escalation.id}/claim", headers=auth(boris))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_resolved"


def test_operator_cannot_resolve_after_claim_expired(db_session):
    anna = make_operator(db_session)
    _, escalation = make_escalated_ticket(db_session)
    now = datetime.now(UTC)
    ops.claim(db_session, escalation.id, anna.id, now=now, ttl=timedelta(minutes=15))

    with pytest.raises(ops.NotClaimedError):
        ops.resolve(
            db_session,
            escalation.id,
            anna.id,
            action=ops.OperatorActionType.CONFIRM,
            final_text=None,
            now=now + timedelta(minutes=16),
        )
