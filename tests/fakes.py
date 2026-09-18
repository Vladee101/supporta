"""Тестовые двойники внешних зависимостей (LLM, ретривер).

Нужны, чтобы пайплайн тестировался без провайдера, без сети и без Postgres:
всё, что в проде приходит извне, здесь задаётся явно.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID, uuid4

from app.domain.enums import ConfidenceSource
from app.services.llm import LabelProbabilities
from app.services.retrieval import RetrievalResult, RetrievedChunk


class FakeLLMClient:
    """LLM с заранее заданным распределением по меткам."""

    def __init__(
        self,
        probabilities: Mapping[str, float],
        *,
        source: ConfidenceSource = ConfidenceSource.LOGPROBS,
        completion: str = "Ответ на основе документов.",
        model_id: str = "fake-llm-v1",
    ) -> None:
        self._probabilities = dict(probabilities)
        self._source = source
        self._completion = completion
        self.model_id = model_id
        self.seen_prompts: list[tuple[str, str]] = []

    def classify(self, system: str, text: str, labels: Sequence[str]) -> LabelProbabilities:
        self.seen_prompts.append((system, text))
        return LabelProbabilities(
            probabilities=self._probabilities,
            source=self._source,
            model_id=self.model_id,
            reasoning="fake",
        )

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        self.seen_prompts.append((system, user))
        return self._completion


def make_chunk(
    slug: str = "delivery-terms",
    score: float = 0.9,
    rank: int = 1,
    content: str = "Доставка по Москве занимает 1-2 рабочих дня.",
    version_id: UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        document_version_id=version_id or uuid4(),
        slug=slug,
        title="Сроки доставки",
        content=content,
        rank=rank,
        relevance_score=score,
    )


class StubRetriever:
    """Ретривер с фиксированной выдачей. Запоминает запросы для проверок PII."""

    def __init__(self, chunks: tuple[RetrievedChunk, ...] = ()) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, session, query: str) -> RetrievalResult:
        self.queries.append(query)
        confidence = max((chunk.relevance_score for chunk in self._chunks), default=None)
        return RetrievalResult(chunks=self._chunks, rag_confidence=confidence)


class StubClassifier:
    """Классификатор с заранее заданным результатом."""

    model_id = "stub-classifier"

    def __init__(self, result) -> None:
        self._result = result
        self.seen: list[str] = []

    def classify(self, text: str):
        self.seen.append(text)
        return self._result
