"""Расчёт embedding'ов для версий документов, у которых их ещё нет.

Провайдер выбирается флагом `--provider`:

* `auto` (по умолчанию) - bge-m3, если он установлен, иначе хеширующий;
* `bge` - только bge-m3 (ADR-006). Первый запуск скачивает ~2 ГБ весов,
  поэтому зависимость вынесена в extra `embeddings`;
* `hashing` - хеширующий bag-of-words. Он **не** заменяет bge-m3 по качеству,
  но позволяет поднять рабочий пайплайн и прогнать eval на машине без модели
  и без сети. Метрики RAG на нём читаются как нижняя граница.

    pip install -e ".[embeddings]"
    python -m scripts.index_kb --provider bge

Скрипт идемпотентен: обрабатывает только версии с `embedding IS NULL`. При смене
провайдера индекс нужно сбросить явно, флагом `--reindex`: иначе в таблице
окажутся векторы из разных пространств, а сравнивать их между собой нельзя.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select, update

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.db.models import EMBEDDING_DIM, KbDocumentVersion
from app.services.embeddings import (
    BgeM3EmbeddingProvider,
    EmbeddingProvider,
    HashingEmbeddingProvider,
)

BATCH_SIZE = 16


def _build_provider(name: str) -> EmbeddingProvider:
    if name == "hashing":
        return HashingEmbeddingProvider()
    if name == "bge":
        return BgeM3EmbeddingProvider()
    try:
        return BgeM3EmbeddingProvider()
    except RuntimeError:
        print("bge-m3 недоступен, используется хеширующий провайдер (качество ниже)")
        return HashingEmbeddingProvider()


def main() -> None:
    # Windows-консоль по умолчанию отдаёт cp1252 и роняет вывод на кириллице.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Индексация базы знаний")
    parser.add_argument("--provider", choices=("auto", "bge", "hashing"), default="auto")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="сбросить существующие embedding'и (обязательно при смене провайдера)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if settings.embedding_dim != EMBEDDING_DIM:
        sys.exit(
            f"embedding_dim в конфиге ({settings.embedding_dim}) не совпадает со схемой "
            f"({EMBEDDING_DIM}): смена модели требует миграции и переиндексации"
        )

    provider = _build_provider(args.provider)

    with get_session_factory()() as session:
        if args.reindex:
            session.execute(
                update(KbDocumentVersion).values(embedding=None, embedding_model=None)
            )
            session.commit()
            print("существующие embedding'и сброшены")

        pending = list(
            session.scalars(
                select(KbDocumentVersion).where(KbDocumentVersion.embedding.is_(None))
            )
        )
        if not pending:
            print("нечего индексировать: у всех версий есть embedding")
            return

        print(f"версий к индексации: {len(pending)}; провайдер: {provider.model_id}")

        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            # Векторы нормированы провайдером: relevance_score определён как
            # 1 - cosine_distance и должен быть сопоставим между документами.
            vectors = provider.encode([version.content for version in batch])
            for version, vector in zip(batch, vectors, strict=True):
                version.embedding = vector
                version.embedding_model = provider.model_id
            session.commit()
            print(f"  проиндексировано {min(start + BATCH_SIZE, len(pending))}/{len(pending)}")

    print("готово")


if __name__ == "__main__":
    main()
