"""tickets.scrubbed_at for retention

Отметка о том, что retention вычистил персональные данные тикета (NFR4).
Делает задачу идемпотентной и оставляет след в самой строке тикета.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tickets", sa.Column("scrubbed_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("tickets", "scrubbed_at")
