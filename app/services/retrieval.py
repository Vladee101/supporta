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

Гибридный режим (ADR-010): векторная и полнотекстовая ветки дают по списку
кандидатов, списки сливаются по RRF (Reciprocal Rank Fusion). Сливаются ранги, а
не оценки: косинус и ts_rank живут на несопоставимых шкалах. Полнотекстовая ветка
ловит то, что эмбеддинги размывают: номера, коды ошибок, редкие термины.

`rag_confidence` в гибриде не меняет смысла - это максимум косинусной близости
среди документов, попавших в итоговый top-k. Оценка RRF в порог не идёт: порог
0.7 откалиброван на косинусе. Документ, найденный только по словам, но далёкий
по смыслу, поэтому не поднимает уверенность до автоответа - консервативная сторона.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isnan
from typing import Literal
from uuid import UUID

from sqlalchemy import Text, cast, func, literal_column, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import SEARCH_CONFIG, KbDocument, KbDocumentVersion
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


#: Константа RRF из исходной статьи (Cormack et al., 2009): сглаживает вклад
#: верхних рангов, чтобы первое место одной ветки не перевешивало согласие обеих.
RRF_K = 60

RetrievalMode = Literal["vector", "hybrid"]


def rrf_fuse(rankings: Sequence[Sequence[UUID]], k: int = RRF_K) -> list[UUID]:
    """Слить ранжированные списки: score = Σ 1 / (k + rank) по спискам с этим id.

    При равенстве оценок выше тот, у кого лучший ранг в каком-либо списке, затем -
    по id: порядок выдачи должен быть детерминированным, он пишется в трейс.
    """
    scores: dict[UUID, float] = {}
    best_rank: dict[UUID, int] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    return sorted(scores, key=lambda key: (-scores[key], best_rank[key], str(key)))


class Retriever:
    def __init__(
        self,
        provider: EmbeddingProvider,
        top_k: int | None = None,
        *,
        mode: RetrievalMode | None = None,
        candidates: int | None = None,
    ) -> None:
        settings = get_settings()
        self._provider = provider
        self._top_k = top_k or settings.rag_top_k
        self._mode: RetrievalMode = mode or settings.rag_retrieval_mode
        self._candidates = max(candidates or settings.rag_candidates, self._top_k)

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
        base = (
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
        )

        limit = self._top_k if self._mode == "vector" else self._candidates
        vector_rows = _defined(session.execute(base.order_by(distance).limit(limit)).all())

        if self._mode == "vector":
            rows = vector_rows[: self._top_k]
        else:
            # Слова запроса объединяются через ИЛИ: у вопроса клиента редко
            # совпадают с документом все слова сразу, а plainto_tsquery требует
            # все (И). Стоп-слова и морфологию снимает конфигурация `russian`.
            tsquery = func.to_tsquery(
                SEARCH_CONFIG,
                func.replace(cast(func.plainto_tsquery(SEARCH_CONFIG, query), Text), " & ", " | "),
            )
            # Заголовок - самый плотный по смыслу текст документа («Режим работы
            # поддержки»), поэтому ищется вместе с текстом и с весом A. Он живёт
            # в документе и меняется без новой версии, так что складывается здесь,
            # а не хранится в search_vector версии: смена заголовка видна поиску сразу.
            document_vector = func.setweight(
                func.to_tsvector(SEARCH_CONFIG, KbDocument.title),
                literal_column("'A'"),  # тип "char": параметр VARCHAR Postgres не примет
            ).op("||")(KbDocumentVersion.search_vector)
            lexical_rows = _defined(
                session.execute(
                    base.where(document_vector.op("@@")(tsquery))
                    .order_by(
                        func.ts_rank_cd(document_vector, tsquery).desc(),
                        KbDocumentVersion.id,
                    )
                    .limit(limit)
                ).all()
            )
            by_id = {row.id: row for row in (*vector_rows, *lexical_rows)}
            fused = rrf_fuse([[row.id for row in vector_rows], [row.id for row in lexical_rows]])
            rows = [by_id[key] for key in fused[: self._top_k]]

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
        )

        rag_confidence = max((chunk.relevance_score for chunk in chunks), default=None)
        return RetrievalResult(chunks=chunks, rag_confidence=rag_confidence)


def _defined(rows: Sequence) -> list:
    """Отбросить строки с неопределённым расстоянием (нулевой вектор документа):
    они не должны ни попасть в выдачу, ни стать rag_confidence."""
    return [row for row in rows if not isnan(float(row.distance))]
