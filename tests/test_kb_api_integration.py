"""KB admin (UC7): версии, асинхронная индексация, soft-delete, доступ только admin."""

from __future__ import annotations

import contextlib
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.kb import get_indexing_session_factory
from app.core.auth import issue_token
from app.db.base import get_session
from app.db.models import KbDocument, KbDocumentVersion
from app.domain.enums import OperatorRole
from app.kb.indexing import index_version
from app.main import app
from app.services.embeddings import HashingEmbeddingProvider
from app.services.retrieval import Retriever
from tests.escalation_fixtures import make_operator

pytestmark = pytest.mark.integration

PROVIDER = HashingEmbeddingProvider()


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_session] = lambda: db_session
    # Фоновая индексация идёт в той же тестовой транзакции; сессию не закрываем.
    app.dependency_overrides[get_indexing_session_factory] = lambda: (
        lambda: contextlib.nullcontext(db_session)
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin(db_session):
    operator = make_operator(db_session, role=OperatorRole.ADMIN)
    return {"Authorization": f"Bearer {issue_token(operator.id, OperatorRole.ADMIN)}"}


@pytest.fixture(autouse=True)
def _hashing_provider(monkeypatch):
    # Роутер берёт провайдер напрямую из фабрики, а не через Depends.
    monkeypatch.setattr("app.api.kb.get_embedding_provider", lambda: PROVIDER)


def create(client, admin, slug="promo-rules", content="Промокоды не суммируются."):
    return client.post(
        "/api/v1/kb/documents",
        json={"slug": slug, "title": "Правила промокодов", "content": content},
        headers=admin,
    )


def search(db_session, query):
    return [chunk.slug for chunk in Retriever(PROVIDER, top_k=5).retrieve(db_session, query).chunks]


def test_operator_role_cannot_manage_kb(client, db_session):
    operator = make_operator(db_session)
    headers = {"Authorization": f"Bearer {issue_token(operator.id, OperatorRole.OPERATOR)}"}
    response = client.get("/api/v1/kb/documents", headers=headers)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_created_document_becomes_searchable_after_indexing(client, admin, db_session):
    """AC UC7: документ доступен для RAG без деплоя - после индексации."""
    response = create(client, admin)

    assert response.status_code == 201
    document_id = uuid.UUID(response.json()["id"])
    # Ответ отдан до индексации: версия создана, но ещё не текущая.
    assert response.json()["indexing"] is True
    assert response.json()["current_version"] is None

    # TestClient выполнил BackgroundTasks после ответа.
    document = db_session.get(KbDocument, document_id)
    assert document.current_version_id is not None
    assert "promo-rules" in search(db_session, "промокоды суммируются")


def test_duplicate_slug_is_409(client, admin):
    create(client, admin)
    response = create(client, admin)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "slug_taken"


def test_edit_creates_version_and_keeps_history(client, admin, db_session):
    document_id = create(client, admin).json()["id"]

    response = client.put(
        f"/api/v1/kb/documents/{document_id}",
        json={"content": "Промокоды не суммируются и не действуют на уценку."},
        headers=admin,
    )
    versions = client.get(f"/api/v1/kb/documents/{document_id}/versions", headers=admin).json()

    assert response.json()["latest_version"] == 2
    assert [v["version"] for v in versions["items"]] == [1, 2]
    assert all(v["indexed"] for v in versions["items"])
    document = db_session.get(KbDocument, uuid.UUID(document_id))
    assert db_session.get(KbDocumentVersion, document.current_version_id).version == 2


def test_title_only_edit_does_not_create_version(client, admin):
    document_id = create(client, admin).json()["id"]
    response = client.put(
        f"/api/v1/kb/documents/{document_id}", json={"title": "Промокоды"}, headers=admin
    )
    assert response.json()["latest_version"] == 1
    assert response.json()["title"] == "Промокоды"


def test_old_version_indexed_late_does_not_win(db_session):
    """Указатель двигается только вперёд, иначе поздняя индексация откатит текст."""
    document = KbDocument(slug="race", title="Гонка")
    db_session.add(document)
    db_session.flush()
    v1 = KbDocumentVersion(document_id=document.id, version=1, content="старый текст")
    v2 = KbDocumentVersion(document_id=document.id, version=2, content="новый текст")
    db_session.add_all([v1, v2])
    db_session.flush()

    assert index_version(db_session, v2.id, PROVIDER) is True
    assert index_version(db_session, v1.id, PROVIDER) is False
    assert db_session.get(KbDocument, document.id).current_version_id == v2.id


def test_unfinished_reindex_keeps_previous_version_searchable(admin, db_session):
    """Пока новая версия не проиндексирована, агент отвечает по предыдущей."""
    document = KbDocument(slug="stable", title="Стабильный")
    db_session.add(document)
    db_session.flush()
    v1 = KbDocumentVersion(document_id=document.id, version=1, content="Гарантия 12 месяцев.")
    db_session.add(v1)
    db_session.flush()
    index_version(db_session, v1.id, PROVIDER)

    db_session.add(KbDocumentVersion(document_id=document.id, version=2, content="черновик"))
    db_session.flush()

    assert "stable" in search(db_session, "гарантия месяцев")


def test_delete_removes_from_rag_but_keeps_versions(client, admin, db_session):
    document_id = create(client, admin).json()["id"]
    assert "promo-rules" in search(db_session, "промокоды суммируются")

    response = client.delete(f"/api/v1/kb/documents/{document_id}", headers=admin)

    assert response.status_code == 200
    assert "promo-rules" not in search(db_session, "промокоды суммируются")
    assert client.get(f"/api/v1/kb/documents/{document_id}", headers=admin).status_code == 404
    versions = client.get(f"/api/v1/kb/documents/{document_id}/versions", headers=admin)
    assert len(versions.json()["items"]) == 1


def test_vectors_from_other_model_are_not_compared(db_session, indexed_kb):
    """Векторы разных провайдеров лежат в разных пространствах - в выдачу не попадают."""
    for version in indexed_kb:
        version.embedding_model = "some-other-model"
    db_session.flush()
    assert search(db_session, "способы оплаты") == []


def test_listing_is_paginated_by_slug(client, admin):
    for slug in ("a-doc", "b-doc", "c-doc"):
        create(client, admin, slug=slug, content=f"Документ {slug}")

    first = client.get("/api/v1/kb/documents", params={"limit": 2}, headers=admin).json()
    second = client.get(
        "/api/v1/kb/documents", params={"limit": 2, "cursor": first["next_cursor"]}, headers=admin
    ).json()

    assert [d["slug"] for d in first["items"]] == ["a-doc", "b-doc"]
    assert [d["slug"] for d in second["items"]] == ["c-doc"]
    assert second["next_cursor"] is None
