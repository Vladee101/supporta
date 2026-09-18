"""Загрузка базы знаний из data/kb_seed.json.

Идемпотентен: повторный запуск не плодит документы и не создаёт версию,
если контент не изменился. Изменённый контент даёт новую версию (ADR-011) -
старая остаётся, потому что на неё ссылаются исторические rag_retrievals.

Embedding здесь не считается: индексация асинхронна (AC из UC7), её делает
scripts/index_kb.py. До появления embedding'а версия не участвует в RAG.

    python -m scripts.seed_kb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.db.base import get_session_factory
from app.db.models import KbDocument, KbDocumentVersion

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "kb_seed.json"


def main() -> None:
    # Windows-консоль по умолчанию отдаёт cp1252 и роняет вывод на кириллице.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    documents = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    created = updated = unchanged = 0

    with get_session_factory()() as session:
        for item in documents:
            slug: str = item["slug"]
            content: str = item["content"]

            document = session.scalar(select(KbDocument).where(KbDocument.slug == slug))
            if document is None:
                document = KbDocument(slug=slug, title=item["title"])
                session.add(document)
                session.flush()
                version_number = 1
                created += 1
            else:
                current = session.get(KbDocumentVersion, document.current_version_id)
                if current is not None and current.content == content:
                    unchanged += 1
                    continue
                version_number = (current.version + 1) if current else 1
                document.title = item["title"]
                document.deleted_at = None
                updated += 1

            version = KbDocumentVersion(
                document_id=document.id, version=version_number, content=content
            )
            session.add(version)
            session.flush()
            document.current_version_id = version.id

        session.commit()

    print(
        f"документов в файле: {len(documents)}; "
        f"создано: {created}, новых версий: {updated}, без изменений: {unchanged}"
    )
    print("следующий шаг: python -m scripts.index_kb (расчёт embedding'ов)")


if __name__ == "__main__":
    main()
