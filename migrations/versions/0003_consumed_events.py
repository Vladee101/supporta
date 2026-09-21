"""consumed events inbox

Inbox consumer'а эскалаций: дедупликация повторных доставок по ключу
идемпотентности из outbox (ADR-007).

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "consumed_events",
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("consumer", sa.String(length=64), nullable=False),
        sa.Column(
            "consumed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("idempotency_key"),
    )


def downgrade() -> None:
    op.drop_table("consumed_events")
