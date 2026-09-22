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


# --- гибридный поиск (ADR-010) ----------------------------------------------


def _retitle(db_session, versions, slug: str, title: str) -> None:
    for version in versions:
        document = db_session.get(KbDocument, version.document_id)
        if document.slug == slug:
            document.title = title
    db_session.flush()


def test_lexical_match_on_title_reaches_top_in_hybrid(db_session, indexed_kb):
    """Слово есть только в заголовке: эмбеддинг текста его не видит, полнотекстовая ветка - да."""
    _retitle(db_session, indexed_kb, "app-crash-on-start", "Холодильники: гарантийный ремонт")

    result = Retriever(HashingEmbeddingProvider(), top_k=1, mode="hybrid").retrieve(
        db_session, "холодильник"
    )
    assert [chunk.slug for chunk in result.chunks] == ["app-crash-on-start"]


def test_title_change_is_searchable_without_reindex(db_session, indexed_kb):
    """Заголовок складывается в запросе, а не хранится в версии: переиндексация не нужна."""
    version = next(
        v for v in indexed_kb if db_session.get(KbDocument, v.document_id).slug == "return-policy"
    )
    embedding_before = list(version.embedding)

    _retitle(db_session, indexed_kb, "return-policy", "Возврат самовара")
    result = Retriever(HashingEmbeddingProvider(), top_k=1, mode="hybrid").retrieve(
        db_session, "самовар"
    )

    assert result.chunks[0].slug == "return-policy"
    assert list(version.embedding) == embedding_before  # версия и её вектор не тронуты


def test_rag_confidence_stays_cosine_in_hybrid(db_session, indexed_kb):
    """Порог 0.7 откалиброван на косинусе: оценка RRF не должна подменять rag_confidence."""
    query = "Какие способы оплаты доступны?"
    vector = Retriever(HashingEmbeddingProvider(), top_k=4, mode="vector").retrieve(
        db_session, query
    )
    hybrid = Retriever(HashingEmbeddingProvider(), top_k=4, mode="hybrid").retrieve(
        db_session, query
    )

    cosine = {chunk.slug: chunk.relevance_score for chunk in vector.chunks}
    assert {chunk.slug: chunk.relevance_score for chunk in hybrid.chunks} == cosine
    assert hybrid.rag_confidence == max(cosine.values())


def test_query_of_stop_words_does_not_break_hybrid(db_session, indexed_kb):
    """plainto_tsquery из одних стоп-слов пуст - лексическая ветка просто ничего не находит."""
    result = Retriever(HashingEmbeddingProvider(), top_k=3, mode="hybrid").retrieve(
        db_session, "и в на с по"
    )
    assert all(0.0 <= chunk.relevance_score <= 1.0 for chunk in result.chunks)
