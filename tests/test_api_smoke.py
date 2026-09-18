"""Smoke-тесты каркаса API. БД для них не нужна: эндпоинты этапа 1-2 к ней не ходят."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_settings, thresholds
from app.domain.decision import RULES
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_decision_table_endpoint_exposes_effective_rules_and_thresholds():
    """Диагностический эндпоинт должен показывать реально действующую конфигурацию."""
    response = client.get("/api/v1/_internal/decision-table")
    assert response.status_code == 200
    body = response.json()

    assert [rule["rule_id"] for rule in body["rules"]] == [rule.rule_id for rule in RULES]
    # Сравнение с действующей конфигурацией, а не с константами: иначе тест
    # зависел бы от .env разработчика (где порог RAG откалиброван под провайдер).
    effective = thresholds()
    assert body["thresholds"]["class_confidence"] == effective.class_confidence
    assert body["thresholds"]["rag_confidence"] == effective.rag_confidence
    assert body["embedding_model"] == get_settings().embedding_model
