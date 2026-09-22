"""SQLAlchemy-модели. Соответствуют ERD из design document один в один.

Ключевые решения схемы, которые легко потерять при чтении кода:

* `classifications` - история по итерациям, а не колонки в `tickets`:
  UC8 требует показать обе итерации отдельно;
* `kb_document_versions` + `rag_retrievals.chunk_snapshot` - аудит показывает
  текст, который агент видел в момент решения, а не текущий (ADR-011);
* `kb_documents.deleted_at` - soft-delete: физическое удаление обрушило бы
  историческую трассируемость (NFR3);
* `outbox_events.idempotency_key` - дедупликация на стороне consumer'а (ADR-007).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.enums import (
    Category,
    ConfidenceSource,
    EscalationReason,
    EscalationStatus,
    MessageSender,
    OperatorActionType,
    OperatorRole,
    RiskLevel,
    TicketStatus,
)

#: Размерность bge-m3 (ADR-006). Меняется только вместе с моделью и переиндексацией.
EMBEDDING_DIM = 1024
#: Конфигурация полнотекстового поиска: морфология русского (стемминг snowball).
SEARCH_CONFIG = "russian"


class Base(DeclarativeBase):
    pass


def _enum_check(column: str, enum_cls: type) -> CheckConstraint:
    """CHECK по значениям python-перечисления.

    Вместо native PG enum: добавление категории не требует ALTER TYPE в миграции,
    а недопустимое значение всё равно не пройдёт в базу.
    """
    values = ", ".join(f"'{member.value}'" for member in enum_cls)
    return CheckConstraint(
        f"{column} IN ({values})", name=f"ck_{column}_{enum_cls.__name__.lower()}"
    )


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Идентификатор обращения в канале. Вместе с channel даёт идемпотентность приёма.
    external_id: Mapped[str | None] = mapped_column(String(128))
    client_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=TicketStatus.NEW)
    #: Денормализованная копия последней классификации; источник истины - classifications.
    category: Mapped[str | None] = mapped_column(String(32))
    class_confidence: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(16))
    clarification_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_client_reply_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Когда retention вычистил персональные данные тикета (NFR4). Строка тикета
    #: при этом остаётся: на ней держатся audit_log и метрики.
    scrubbed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list[Message]] = relationship(back_populates="ticket")
    classifications: Mapped[list[Classification]] = relationship(back_populates="ticket")
    escalations: Mapped[list[Escalation]] = relationship(back_populates="ticket")

    __table_args__ = (
        UniqueConstraint("channel", "external_id", name="uq_tickets_channel_external_id"),
        _enum_check("status", TicketStatus),
        _enum_check("category", Category),
        _enum_check("risk_level", RiskLevel),
        CheckConstraint("clarification_count >= 0", name="ck_tickets_clarification_count"),
        Index("ix_tickets_status_created_at", "status", "created_at"),
    )


class Classification(Base):
    """Одна итерация классификации. Повторная не затирает предыдущую (UC8)."""

    __tablename__ = "classifications"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    #: Как получен confidence - шкала не переносится между провайдерами.
    confidence_source: Mapped[str | None] = mapped_column(String(32))
    model_id: Mapped[str | None] = mapped_column(String(128))
    reasoning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    ticket: Mapped[Ticket] = relationship(back_populates="classifications")

    __table_args__ = (
        UniqueConstraint("ticket_id", "iteration", name="uq_classifications_ticket_iteration"),
        _enum_check("category", Category),
        _enum_check("risk_level", RiskLevel),
        _enum_check("confidence_source", ConfidenceSource),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    sender: Mapped[str] = mapped_column(String(16), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: Версия с замаскированным PII - именно она уходит в LLM (NFR4).
    content_redacted: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    ticket: Mapped[Ticket] = relationship(back_populates="messages")

    __table_args__ = (
        _enum_check("sender", MessageSender),
        Index("ix_messages_ticket_created_at", "ticket_id", "created_at"),
    )


class AuditLog(Base):
    """Append-only. Полный набор для NFR3: правило, оба confidence, reasoning."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    actor: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Сработавшая строка decision table: R1..R9, R7c, R-default.
    rule_id: Mapped[str | None] = mapped_column(String(16))
    class_confidence: Mapped[float | None] = mapped_column(Float)
    rag_confidence: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    reasoning: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (Index("ix_audit_log_ticket_created_at", "ticket_id", "created_at"),)


class KbDocument(Base):
    __tablename__ = "kb_documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "kb_document_versions.id",
            use_alter=True,
            name="fk_kb_documents_current_version",
        ),
    )
    #: Soft-delete: физическое удаление обрушило бы аудит прошлых решений (UC7 AC).
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    versions: Mapped[list[KbDocumentVersion]] = relationship(
        back_populates="document", foreign_keys="KbDocumentVersion.document_id"
    )


class KbDocumentVersion(Base):
    """Версия контента с собственным embedding'ом.

    Документ участвует в RAG только когда у версии посчитан embedding -
    отсюда nullable-колонка и фильтр в поиске (асинхронная индексация, UC7 AC).
    """

    __tablename__ = "kb_document_versions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kb_documents.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    #: Лексическая половина гибридного поиска (ADR-010). Считается самим Postgres
    #: из текста версии: не зависит от embedding-провайдера и асинхронной
    #: индексации, версия ищется по словам сразу после сохранения.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(f"to_tsvector('{SEARCH_CONFIG}', content)", persisted=True)
    )
    created_at: Mapped[datetime] = _created_at()

    document: Mapped[KbDocument] = relationship(
        back_populates="versions", foreign_keys=[document_id]
    )

    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_kb_document_versions_doc_version"),
        CheckConstraint("version >= 1", name="ck_kb_document_versions_version"),
        # HNSW по косинусу: векторный поиск должен укладываться в 0.1 сек из
        # бюджета задержки (NFR1). Косинус - потому что rag_confidence определён
        # как 1 - cosine_distance.
        Index(
            "ix_kb_document_versions_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_kb_document_versions_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )


class RagRetrieval(Base):
    """Тикет ↔ версия документа. Снапшот чанка - на случай смены чанкинга (ADR-010)."""

    __tablename__ = "rag_retrievals"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    #: RESTRICT: история ретривала переживает удаление документа (UC7 AC).
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kb_document_versions.id", ondelete="RESTRICT"), nullable=False
    )
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    chunk_snapshot: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_rag_retrievals_ticket_iteration", "ticket_id", "iteration"),
        CheckConstraint("rank >= 1", name="ck_rag_retrievals_rank"),
    )


class Operator(Base):
    __tablename__ = "operators"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default=OperatorRole.OPERATOR)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (_enum_check("role", OperatorRole),)


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    operator_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("operators.id"))
    reason: Mapped[str] = mapped_column(String(48), nullable=False)
    #: Правило, приведшее к эскалации - вместе с reason даёт полную причину (NFR3).
    rule_id: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=EscalationStatus.PENDING
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    draft_text: Mapped[str | None] = mapped_column(Text)
    #: Claim оператора (FR10): TTL снимается шедулером.
    locked_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("operators.id"))
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    ticket: Mapped[Ticket] = relationship(back_populates="escalations")

    __table_args__ = (
        _enum_check("reason", EscalationReason),
        _enum_check("status", EscalationStatus),
        Index("ix_escalations_queue", "status", "priority", "created_at"),
    )


class OperatorAction(Base):
    """Diff между черновиком агента и финальным ответом оператора (NFR7)."""

    __tablename__ = "operator_actions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    escalation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("escalations.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    operator_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("operators.id"), nullable=False)
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)
    draft_text: Mapped[str | None] = mapped_column(Text)
    final_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (_enum_check("action_type", OperatorActionType),)


class OutboxEvent(Base):
    """Transactional outbox (ADR-007): пишется в одной транзакции с эскалацией."""

    __tablename__ = "outbox_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Ключ дедупликации для consumer'а: outbox сам по себе не даёт exactly-once.
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    published: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index(
            "ix_outbox_events_unpublished",
            "created_at",
            postgresql_where=text("published = false"),
        ),
    )


class ConsumedEvent(Base):
    """Inbox consumer'а: какие события уже обработаны.

    Outbox даёт доставку at-least-once - поллер может опубликовать событие
    повторно, если упадёт между публикацией и commit'ом отметки `published`.
    Consumer записывает `idempotency_key` в этой таблице в той же транзакции,
    что и изменение состояния; повторная доставка упирается в первичный ключ
    и становится no-op (ADR-007).
    """

    __tablename__ = "consumed_events"

    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    consumer: Mapped[str] = mapped_column(String(64), nullable=False)
    consumed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
