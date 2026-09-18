"""Интеграционные тесты ретривера: pgvector, фильтры выдачи, rag_confidence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.db.models import KbDocument, KbDocumentVersion
from app.services.embeddings import HashingEmbeddingProvider
from app.services.retrieval import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture
def retriever() -> Retriever:
    return Retriever(HashingEmbeddingProvider(), top_k=3)


def test_relevant_document_is_ranked_first(db_session, indexed_kb, retriever):
    result = retriever.retrieve(db_session, "Какие способы оплаты доступны?")
    assert result.chunks[0].slug == "payment-methods"


def test_rag_confidence_is_max_score_not_mean(db_session, indexed_kb, retriever):
    """Определение из «Confidence и пороги» должно совпадать с кодом."""
    result = retriever.retrieve(db_session, "Сроки доставки по Москве")
    scores = [chunk.relevance_score for chunk in result.chunks]

    assert result.rag_confidence == max(scores)
    assert len(scores) > 1
    assert result.rag_confidence != pytest.approx(sum(scores) / len(scores))


def test_top_k_limits_output(db_session, indexed_kb):
    result = Retriever(HashingEmbeddingProvider(), top_k=2).retrieve(db_session, "доставка")
    assert len(result.chunks) == 2


def test_ranks_are_dense_and_ordered(db_session, indexed_kb, retriever):
    result = retriever.retrieve(db_session, "возврат товара")
    assert [chunk.rank for chunk in result.chunks] == [1, 2, 3]
    scores = [chunk.relevance_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_soft_deleted_document_leaves_the_index(db_session, indexed_kb, retriever):
    """UC7: документ исчезает из выдачи, но его версии остаются для аудита."""
    query = "Какие способы оплаты доступны?"
    assert retriever.retrieve(db_session, query).chunks[0].slug == "payment-methods"

    document = db_session.query(KbDocument).filter_by(slug="payment-methods").one()
    document.deleted_at = datetime.now(UTC)
    db_session.flush()

    slugs = [chunk.slug for chunk in retriever.retrieve(db_session, query).chunks]
    assert "payment-methods" not in slugs
    assert db_session.query(KbDocumentVersion).filter_by(document_id=document.id).count() == 1


def test_only_current_version_participates_in_search(db_session, indexed_kb, retriever):
    """Старая версия не должна конкурировать с актуальной."""
    document = db_session.query(KbDocument).filter_by(slug="delivery-terms").one()
    provider = HashingEmbeddingProvider()
    new_content = "Доставка по Москве теперь занимает один рабочий день."
    version = KbDocumentVersion(
        document_id=document.id,
        version=2,
        content=new_content,
        embedding=provider.encode([new_content])[0],
        embedding_model=provider.model_id,
    )
    db_session.add(version)
    db_session.flush()
    document.current_version_id = version.id
    db_session.flush()

    chunks = retriever.retrieve(db_session, "сроки доставки по Москве").chunks
    delivery = [chunk for chunk in chunks if chunk.slug == "delivery-terms"]
    assert len(delivery) == 1
    assert delivery[0].document_version_id == version.id


def test_unindexed_version_is_invisible_to_rag(db_session, indexed_kb, retriever):
    """Индексация асинхронна: версия без embedding в поиске не участвует (UC7)."""
    document = KbDocument(slug="new-doc", title="Новый документ")
    db_session.add(document)
    db_session.flush()
    version = KbDocumentVersion(
        document_id=document.id, version=1, content="Промокоды не суммируются."
    )
    db_session.add(version)
    db_session.flush()
    document.current_version_id = version.id
    db_session.flush()

    slugs = [chunk.slug for chunk in retriever.retrieve(db_session, "промокоды").chunks]
    assert "new-doc" not in slugs


def test_empty_knowledge_base_gives_no_confidence(db_session, retriever):
    result = retriever.retrieve(db_session, "что угодно")
    assert result.is_empty
    assert result.rag_confidence is None
