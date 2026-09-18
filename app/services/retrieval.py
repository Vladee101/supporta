"""RAG-поиск по базе знаний.

`rag_confidence = max(relevance_score)` по top-k, где
`relevance_score = 1 - cosine_distance` - определение зафиксировано в разделе
«Confidence и пороги» и не должно расходиться с кодом: максимум, а не среднее,
потому что одного точно подходящего документа достаточно для ответа.

Ищем только по текущим версиям неудалённых документов с готовым embedding'ом:

* `deleted_at IS NULL` - soft-deleted документ исключён из выдачи, но его версии
  остаются для аудита прошлых решений (ADR-011);
* `current_version_id = version.id` - старые версии не конкурируют с актуальной;
* `embedding IS NOT NULL` - индексация асинхронна, непроиндексированная версия
  в RAG не участвует (AC из UC7).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import KbDocument, KbDocumentVersion
from app.services.embeddings import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: UUID
    document_version_id: UUID
    slug: str
    title: str
    content: str
    rank: int
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunks: tuple[RetrievedChunk, ...]
    #: None, если выдача пуста: отсутствие измерения ниже порога по определению
    #: (Decision Engine трактует None как «ниже порога»).
    rag_confidence: float | None

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Retriever:
    def __init__(self, provider: EmbeddingProvider, top_k: int | None = None) -> None:
        self._provider = provider
        self._top_k = top_k or get_settings().rag_top_k

    def retrieve(self, session: Session, query: str) -> RetrievalResult:
        query_vector = self._provider.encode_query(query)

        # Вырожденный запрос («???», пустая строка, только стоп-символы) даёт
        # нулевой вектор, а косинусное расстояние до него не определено -
        # pgvector возвращает NaN. Это не ошибка поиска, а отсутствие сигнала:
        # выдача пуста, rag_confidence = None, и Decision Engine трактует это
        # как «ниже порога».
        if not any(query_vector):
            return RetrievalResult(chunks=(), rag_confidence=None)

        distance = KbDocumentVersion.embedding.cosine_distance(query_vector)

        rows = session.execute(
            select(
                KbDocumentVersion.id,
                KbDocumentVersion.document_id,
                KbDocumentVersion.content,
                KbDocument.slug,
                KbDocument.title,
                distance.label("distance"),
            )
            .join(KbDocument, KbDocument.id == KbDocumentVersion.document_id)
            .where(
                KbDocument.deleted_at.is_(None),
                KbDocument.current_version_id == KbDocumentVersion.id,
                KbDocumentVersion.embedding.is_not(None),
                # Векторы разных моделей лежат в разных пространствах, и
                # косинус между ними ничего не значит. Версия, проиндексированная
                # другим провайдером, в выдачу не попадает - её нужно
                # переиндексировать (scripts.index_kb --reindex).
                KbDocumentVersion.embedding_model == self._provider.model_id,
            )
            .order_by(distance)
            .limit(self._top_k)
        ).all()

        chunks = tuple(
            RetrievedChunk(
                document_id=row.document_id,
                document_version_id=row.id,
                slug=row.slug,
                title=row.title,
                content=row.content,
                rank=rank,
                # Косинусное расстояние в [0, 2]; для нормированных векторов
                # с неотрицательными координатами практически всегда [0, 1].
                relevance_score=round(max(0.0, min(1.0, 1.0 - float(row.distance))), 4),
            )
            for rank, row in enumerate(rows, start=1)
            # Строка с неопределённым расстоянием (нулевой вектор документа)
            # не должна попасть в выдачу и тем более стать rag_confidence.
            if not isnan(float(row.distance))
        )

        rag_confidence = max((chunk.relevance_score for chunk in chunks), default=None)
        return RetrievalResult(chunks=chunks, rag_confidence=rag_confidence)
