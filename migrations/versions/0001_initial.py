"""initial schema

Схема из раздела «Схема данных (ERD)» design document, ревизия 2:
classifications (история итераций), kb_document_versions (версионирование KB),
soft-delete документов, claim эскалаций, outbox с ключом идемпотентности.

Revision ID: 0001
Revises:
Create Date: 2026-09-18
"""
from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector - предусловие схемы: без расширения не создастся тип vector (ADR-006).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('kb_documents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('slug', sa.String(length=128), nullable=False),
    sa.Column('title', sa.String(length=512), nullable=False),
    sa.Column('current_version_id', sa.UUID(), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('slug')
    )
    op.create_table('operators',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('email', sa.String(length=256), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("role IN ('operator', 'admin')", name='ck_role_operatorrole'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    op.create_table('tickets',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('channel', sa.String(length=32), nullable=False),
    sa.Column('external_id', sa.String(length=128), nullable=True),
    sa.Column('client_id', sa.String(length=128), nullable=True),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('category', sa.String(length=32), nullable=True),
    sa.Column('class_confidence', sa.Float(), nullable=True),
    sa.Column('risk_level', sa.String(length=16), nullable=True),
    sa.Column('clarification_count', sa.Integer(), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('last_client_reply_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("category IN ('faq', 'order_status', 'complaint', 'refund', 'tech_issue', 'unclassified')", name='ck_category_category'),
    sa.CheckConstraint("risk_level IN ('low', 'medium', 'high')", name='ck_risk_level_risklevel'),
    sa.CheckConstraint("status IN ('new', 'classified', 'awaiting_clarification', 'escalated_standard', 'escalated_priority', 'pending_operator', 'in_progress', 'resolved_auto', 'resolved_by_operator')", name='ck_status_ticketstatus'),
    sa.CheckConstraint('clarification_count >= 0', name='ck_tickets_clarification_count'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('channel', 'external_id', name='uq_tickets_channel_external_id')
    )
    op.create_table('audit_log',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('ticket_id', sa.UUID(), nullable=False),
    sa.Column('actor', sa.String(length=32), nullable=False),
    sa.Column('action', sa.String(length=64), nullable=False),
    sa.Column('rule_id', sa.String(length=16), nullable=True),
    sa.Column('class_confidence', sa.Float(), nullable=True),
    sa.Column('rag_confidence', sa.Float(), nullable=True),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('reasoning', sa.Text(), nullable=True),
    sa.Column('trace_id', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('classifications',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('ticket_id', sa.UUID(), nullable=False),
    sa.Column('iteration', sa.Integer(), nullable=False),
    sa.Column('category', sa.String(length=32), nullable=False),
    sa.Column('risk_level', sa.String(length=16), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('confidence_source', sa.String(length=32), nullable=True),
    sa.Column('model_id', sa.String(length=128), nullable=True),
    sa.Column('reasoning', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("category IN ('faq', 'order_status', 'complaint', 'refund', 'tech_issue', 'unclassified')", name='ck_category_category'),
    sa.CheckConstraint("confidence_source IN ('logprobs', 'k_sampling')", name='ck_confidence_source_confidencesource'),
    sa.CheckConstraint("risk_level IN ('low', 'medium', 'high')", name='ck_risk_level_risklevel'),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('ticket_id', 'iteration', name='uq_classifications_ticket_iteration')
    )
    op.create_table('escalations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('ticket_id', sa.UUID(), nullable=False),
    sa.Column('operator_id', sa.UUID(), nullable=True),
    sa.Column('reason', sa.String(length=48), nullable=False),
    sa.Column('rule_id', sa.String(length=16), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('draft_text', sa.Text(), nullable=True),
    sa.Column('locked_by', sa.UUID(), nullable=True),
    sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("reason IN ('client_requested', 'high_risk_category', 'classification_failed', 'low_class_confidence', 'low_rag_confidence', 'clarification_limit_reached', 'clarification_timeout', 'agent_timeout', 'llm_unavailable', 'decision_table_gap')", name='ck_reason_escalationreason'),
    sa.CheckConstraint("status IN ('pending', 'in_progress', 'resolved')", name='ck_status_escalationstatus'),
    sa.ForeignKeyConstraint(['locked_by'], ['operators.id'], ),
    sa.ForeignKeyConstraint(['operator_id'], ['operators.id'], ),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('kb_document_versions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=True),
    sa.Column('embedding_model', sa.String(length=128), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('version >= 1', name='ck_kb_document_versions_version'),
    sa.ForeignKeyConstraint(['document_id'], ['kb_documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'version', name='uq_kb_document_versions_doc_version')
    )
    op.create_table('messages',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('ticket_id', sa.UUID(), nullable=False),
    sa.Column('sender', sa.String(length=16), nullable=False),
    sa.Column('iteration', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('content_redacted', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("sender IN ('client', 'agent', 'operator')", name='ck_sender_messagesender'),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('outbox_events',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('ticket_id', sa.UUID(), nullable=False),
    sa.Column('event_type', sa.String(length=64), nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('published', sa.Boolean(), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('idempotency_key')
    )
    op.create_table('operator_actions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('escalation_id', sa.UUID(), nullable=False),
    sa.Column('operator_id', sa.UUID(), nullable=False),
    sa.Column('action_type', sa.String(length=16), nullable=False),
    sa.Column('draft_text', sa.Text(), nullable=True),
    sa.Column('final_text', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("action_type IN ('confirm', 'edit', 'reject')", name='ck_action_type_operatoractiontype'),
    sa.ForeignKeyConstraint(['escalation_id'], ['escalations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['operator_id'], ['operators.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('escalation_id')
    )
    op.create_table('rag_retrievals',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('ticket_id', sa.UUID(), nullable=False),
    sa.Column('document_version_id', sa.UUID(), nullable=False),
    sa.Column('iteration', sa.Integer(), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=False),
    sa.Column('relevance_score', sa.Float(), nullable=False),
    sa.Column('chunk_snapshot', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('rank >= 1', name='ck_rag_retrievals_rank'),
    sa.ForeignKeyConstraint(['document_version_id'], ['kb_document_versions.id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tickets_status_created_at', 'tickets', ['status', 'created_at'], unique=False)
    op.create_index('ix_audit_log_ticket_created_at', 'audit_log', ['ticket_id', 'created_at'], unique=False)
    op.create_index('ix_escalations_queue', 'escalations', ['status', 'priority', 'created_at'], unique=False)
    op.create_index('ix_kb_document_versions_embedding_hnsw', 'kb_document_versions', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('ix_messages_ticket_created_at', 'messages', ['ticket_id', 'created_at'], unique=False)
    op.create_index('ix_outbox_events_unpublished', 'outbox_events', ['created_at'], unique=False, postgresql_where=sa.text('published = false'))
    op.create_index('ix_rag_retrievals_ticket_iteration', 'rag_retrievals', ['ticket_id', 'iteration'], unique=False)
    # Круговая ссылка: kb_documents.current_version_id → kb_document_versions.id.
    op.create_foreign_key(
    "fk_kb_documents_current_version",
    "kb_documents",
    "kb_document_versions",
    ["current_version_id"],
    ["id"],
)


def downgrade() -> None:
    # Сначала снимается круговой FK, иначе kb_document_versions не удалить.
    op.drop_constraint("fk_kb_documents_current_version", "kb_documents", type_="foreignkey")
    op.drop_table("rag_retrievals")
    op.drop_table("operator_actions")
    op.drop_table("outbox_events")
    op.drop_table("messages")
    op.drop_table("kb_document_versions")
    op.drop_table("escalations")
    op.drop_table("classifications")
    op.drop_table("audit_log")
    op.drop_table("tickets")
    op.drop_table("operators")
    op.drop_table("kb_documents")
