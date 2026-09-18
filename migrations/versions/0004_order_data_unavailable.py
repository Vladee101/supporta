"""order_data_unavailable escalation reason

Причина эскалации для R10: статус заказа уходит на оператора, пока нет
интеграции с системой заказов.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-19
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "ck_reason_escalationreason"
TABLE = "escalations"

BASE_REASONS = (
    "'client_requested', 'high_risk_category', 'classification_failed', "
    "'low_class_confidence', 'low_rag_confidence', 'clarification_limit_reached', "
    "'clarification_timeout', 'agent_timeout', 'llm_unavailable', 'decision_table_gap'"
)


def upgrade() -> None:
    op.drop_constraint(CONSTRAINT, TABLE, type_="check")
    op.create_check_constraint(
        CONSTRAINT, TABLE, f"reason IN ({BASE_REASONS}, 'order_data_unavailable')"
    )


def downgrade() -> None:
    # Эскалации с новой причиной откат не переживут - это решение оператора БД,
    # поэтому downgrade падает на CHECK, а не удаляет данные молча.
    op.drop_constraint(CONSTRAINT, TABLE, type_="check")
    op.create_check_constraint(CONSTRAINT, TABLE, f"reason IN ({BASE_REASONS})")
