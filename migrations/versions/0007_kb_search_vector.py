"""kb_document_versions.search_vector for hybrid retrieval

Лексическая половина гибридного поиска (ADR-010): генерируемая колонка
tsvector по тексту версии с русской морфологией и GIN-индекс. Триггер
включения гибрида из ADR-010 сработал - Recall@5 на golden set ниже 0.9.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "kb_document_versions"
INDEX = "ix_kb_document_versions_search_vector"


def upgrade() -> None:
    # STORED-колонка заполняется для существующих строк при добавлении:
    # переиндексация базы знаний не нужна.
    op.add_column(
        TABLE,
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('russian', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(INDEX, TABLE, ["search_vector"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index(INDEX, table_name=TABLE)
    op.drop_column(TABLE, "search_vector")
