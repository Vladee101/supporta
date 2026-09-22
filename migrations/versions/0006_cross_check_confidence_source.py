"""cross_check confidence source

Добавляет значение `cross_check` в CHECK на classifications.confidence_source:
уверенность LLM, сверенная с базовой линией (ADR-012), - своя шкала, и в трейсе
она должна быть отличима от сырых logprobs и k-sampling.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "ck_confidence_source_confidencesource"
TABLE = "classifications"


def upgrade() -> None:
    op.drop_constraint(CONSTRAINT, TABLE, type_="check")
    op.create_check_constraint(
        CONSTRAINT,
        TABLE,
        "confidence_source IN ('logprobs', 'k_sampling', 'baseline', 'cross_check')",
    )


def downgrade() -> None:
    # Строки с cross_check-источником не переживут отката: их нужно снять заранее.
    op.execute("DELETE FROM classifications WHERE confidence_source = 'cross_check'")
    op.drop_constraint(CONSTRAINT, TABLE, type_="check")
    op.create_check_constraint(
        CONSTRAINT,
        TABLE,
        "confidence_source IN ('logprobs', 'k_sampling', 'baseline')",
    )
