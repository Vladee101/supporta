"""baseline confidence source

Добавляет значение `baseline` в CHECK на classifications.confidence_source:
у словарной базовой линии своя шкала confidence, и она должна быть отличима
от logprobs в трейсе (NFR3, «Confidence и пороги»).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-18
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "ck_confidence_source_confidencesource"
TABLE = "classifications"


def upgrade() -> None:
    op.drop_constraint(CONSTRAINT, TABLE, type_="check")
    op.create_check_constraint(
        CONSTRAINT,
        TABLE,
        "confidence_source IN ('logprobs', 'k_sampling', 'baseline')",
    )


def downgrade() -> None:
    # Строки с baseline-источником не переживут отката: их нужно снять заранее.
    op.execute("DELETE FROM classifications WHERE confidence_source = 'baseline'")
    op.drop_constraint(CONSTRAINT, TABLE, type_="check")
    op.create_check_constraint(
        CONSTRAINT,
        TABLE,
        "confidence_source IN ('logprobs', 'k_sampling')",
    )
