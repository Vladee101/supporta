"""Провайдеры embedding'ов.

Основной - bge-m3, self-hosted (ADR-006): нет переменной стоимости за вызов и
нет внешнего round-trip в RAG-пути. Цена решения - около 2 ГБ весов, которые
нужно скачать, поэтому зависимость лежит в extra `embeddings`.

Второй провайдер, `HashingEmbeddingProvider`, нужен не для качества, а для
воспроизводимости: пайплайн, интеграционные тесты и прогон eval должны
запускаться на машине без модели и без сети. Он даёт осмысленный лексический
сигнал, но заметно слабее bge-m3 - метрики RAG, снятые на нём, читаются как
нижняя граница, а не как результат системы.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.core.config import get_settings


class EmbeddingProvider(Protocol):
    model_id: str
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Векторы документов. Нормированы: relevance_score = 1 - cosine_distance."""
        ...

    def encode_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"\w{3,}", re.UNICODE)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


#: Длина символьной n-граммы. Слова целиком на русском не совпадают из-за
#: словоизменения («оплаты» против «оплата»), а n-граммы общий корень ловят.
NGRAM = 4


class HashingEmbeddingProvider:
    """Хеширующий мешок признаков: слова целиком плюс символьные n-граммы слов.

    Без n-грамм провайдер бесполезен на русском: запрос «какие способы оплаты»
    и документ со словом «оплата» не пересекаются ни одним токеном. N-граммы
    дают общий корень, а стеммер сюда тащить незачем - это вспомогательный
    провайдер, а не замена bge-m3.

    Каждый признак раскладывается в три измерения (три хеш-функции), чтобы
    снизить влияние коллизий; вес - sublinear tf, повтор не раздувает вектор
    линейно.
    """

    model_id = "hashing-ngram-v1"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or get_settings().embedding_dim

    @staticmethod
    def _features(text: str):
        for word in _TOKEN_RE.findall(text.lower()):
            yield word
            for start in range(max(0, len(word) - NGRAM + 1)):
                yield word[start : start + NGRAM]

    def _vector(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=12).digest()
            for offset in (0, 4, 8):
                index = int.from_bytes(digest[offset : offset + 4], "big") % self.dim
                counts[index] = counts.get(index, 0.0) + 1.0

        vector = [0.0] * self.dim
        for index, count in counts.items():
            vector[index] = 1.0 + math.log(count)
        return _normalize(vector)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)


class BgeM3EmbeddingProvider:
    """bge-m3 через sentence-transformers (ADR-006)."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                'sentence-transformers не установлен: pip install -e ".[embeddings]"'
            ) from exc

        settings = get_settings()
        self.model_id = model_name or settings.embedding_model
        self.dim = settings.embedding_dim
        self._model = SentenceTransformer(self.model_id)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> list[float]:
        return self.encode([text])[0]


def get_embedding_provider(prefer_real: bool = True) -> EmbeddingProvider:
    """bge-m3, если он доступен; иначе - хеширующий провайдер с предупреждением."""
    if prefer_real:
        try:
            return BgeM3EmbeddingProvider()
        except RuntimeError:
            pass
    return HashingEmbeddingProvider()
