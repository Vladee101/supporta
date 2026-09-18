"""Имитация задержки LLM - только для нагрузочного теста (NFR1, NFR8).

Базовая линия (словарный классификатор, шаблонный генератор) отвечает за
миллисекунды. Нагрузочный тест на ней мерил бы не систему, а её пустую
оболочку: 50 одновременных запросов по 5 мс не нагружают ни пул потоков, ни
пул соединений с базой. С провайдером каждый тикет держит поток и, возможно,
соединение по несколько секунд - именно это и нужно проверить.

Задержка берётся равномерно из [0.3 × max, max], где max - p95-бюджет шага
из раздела «Бюджет задержки»: классификация 2.0 сек, генерация 4.5 сек.
Выключено по умолчанию (0 мс); включается только конфигом нагрузочного стенда.
"""

from __future__ import annotations

import random
import time

from app.services.classifier import ClassificationResult, Classifier
from app.services.generation import Draft, ResponseGenerator
from app.services.retrieval import RetrievedChunk

LOWER_FRACTION = 0.3


def _sleep(max_ms: int, rng: random.Random) -> None:
    time.sleep(rng.uniform(max_ms * LOWER_FRACTION, max_ms) / 1000)


class SlowClassifier:
    def __init__(self, inner: Classifier, max_ms: int, seed: int | None = None) -> None:
        self._inner = inner
        self._max_ms = max_ms
        self._rng = random.Random(seed)

    @property
    def model_id(self) -> str:
        return getattr(self._inner, "model_id", "unknown")

    def classify(self, text: str) -> ClassificationResult:
        _sleep(self._max_ms, self._rng)
        return self._inner.classify(text)


class SlowGenerator:
    def __init__(self, inner: ResponseGenerator, max_ms: int, seed: int | None = None) -> None:
        self._inner = inner
        self._max_ms = max_ms
        self._rng = random.Random(seed)

    def generate(self, ticket_text: str, chunks: tuple[RetrievedChunk, ...]) -> Draft:
        _sleep(self._max_ms, self._rng)
        return self._inner.generate(ticket_text, chunks)
