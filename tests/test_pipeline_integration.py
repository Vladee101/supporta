"""Интеграционные тесты пайплайна: что именно оседает в базе после решения агента.

Проверяется не «код отработал», а выполнимость требований по трассируемости:
NFR3 (правило + оба confidence + документы в трейсе), NFR4 (в логи не попадает
PII), UC8 (обе итерации классификации видны отдельно), ADR-007 (эскалация и
событие outbox пишутся вместе).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.agent.graph import TicketGraph
from app.agent.service import AgentService
from app.db.models import (
    AuditLog,
    Classification,
    Escalation,
    Message,
    OutboxEvent,
    RagRetrieval,
    Ticket,
)
from app.domain.decision import PRIORITY_CLIENT_REQUESTED, Thresholds
from app.domain.enums import Action, MessageSender, TicketStatus
from app.services.classifier import BaselineClassifier
from app.services.embeddings import HashingEmbeddingProvider
from app.services.generation import TemplateResponseGenerator
from app.services.retrieval import Retriever

pytestmark = pytest.mark.integration


#: Порог RAG под хеширующий провайдер. Косинусная шкала не переносится между
#: моделями: у hashing-ngram верх выдачи лежит в районе 0.2-0.5, а 0.7 - порог,
#: откалиброванный под bge-m3 (см. «Confidence и пороги»). Использовать здесь
#: продовое значение означало бы проверять не пайплайн, а несовпадение шкал.
HASHING_RAG_THRESHOLD = 0.15


@pytest.fixture
def service() -> AgentService:
    graph = TicketGraph(
        BaselineClassifier(),
        Retriever(HashingEmbeddingProvider(), top_k=3),
        TemplateResponseGenerator(),
        Thresholds(rag_confidence=HASHING_RAG_THRESHOLD),
    )
    return AgentService(graph)


def make_ticket(session, text: str, *, status=TicketStatus.NEW, clarifications=0) -> Ticket:
    ticket = Ticket(
        channel="web",
        external_id=f"ext-{abs(hash(text)) % 10**8}",
        status=status.value,
        clarification_count=clarifications,
    )
    session.add(ticket)
    session.flush()
    session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.CLIENT,
            iteration=clarifications,
            content=text,
        )
    )
    session.flush()
    return ticket


def test_auto_answer_is_persisted_with_full_trace(db_session, indexed_kb, service):
    ticket = make_ticket(db_session, "Подскажите, какие способы оплаты доступны?")
    result = service.process(db_session, ticket)

    assert result.outcome.decision.action is Action.AUTO_ANSWER
    assert ticket.status == TicketStatus.RESOLVED_AUTO
    assert ticket.resolved_at is not None
    assert result.escalation_id is None

    audit = db_session.scalar(select(AuditLog).where(AuditLog.ticket_id == ticket.id))
    assert audit.rule_id == "R4"
    assert audit.class_confidence is not None
    assert audit.rag_confidence is not None
    assert audit.trace_id == result.trace_id
    assert audit.payload["retrieved"], "в трейсе должны быть использованные документы"

    reply = db_session.scalars(
        select(Message).where(Message.ticket_id == ticket.id, Message.sender == "agent")
    ).one()
    assert reply.content


def test_retrievals_are_linked_to_document_versions(db_session, indexed_kb, service):
    """Аудит ссылается на версию, а не на документ: текст мог измениться (ADR-011)."""
    ticket = make_ticket(db_session, "Подскажите, какие способы оплаты доступны?")
    service.process(db_session, ticket)

    retrievals = db_session.scalars(
        select(RagRetrieval).where(RagRetrieval.ticket_id == ticket.id)
    ).all()
    known_versions = {version.id for version in indexed_kb}

    assert retrievals
    assert all(row.document_version_id in known_versions for row in retrievals)
    assert all(row.chunk_snapshot for row in retrievals)


def test_escalation_and_outbox_event_are_written_together(db_session, indexed_kb, service):
    """ADR-007: событие и эскалация - в одной транзакции, иначе гарантии текут."""
    ticket = make_ticket(db_session, "Требую вернуть деньги за бракованный товар")
    result = service.process(db_session, ticket)

    escalation = db_session.get(Escalation, result.escalation_id)
    event = db_session.scalar(select(OutboxEvent).where(OutboxEvent.ticket_id == ticket.id))

    assert escalation.reason == "high_risk_category"
    assert escalation.rule_id == "R2/R3"
    assert event.idempotency_key == f"escalation.created:{escalation.id}"
    assert event.published is False
    assert event.payload["escalation_id"] == str(escalation.id)


def test_high_risk_escalation_carries_no_draft(db_session, indexed_kb, service):
    ticket = make_ticket(db_session, "Пишу претензию: курьер нахамил, коробка повреждена")
    result = service.process(db_session, ticket)

    escalation = db_session.get(Escalation, result.escalation_id)
    assert escalation.draft_text is None


def test_tech_issue_escalation_carries_draft(db_session, indexed_kb, service):
    ticket = make_ticket(
        db_session, "Приложение вылетает при запуске", clarifications=1
    )
    result = service.process(db_session, ticket)

    escalation = db_session.get(Escalation, result.escalation_id)
    assert escalation.rule_id == "R7b"
    assert escalation.draft_text


def test_clarification_increments_counter_and_sets_status(db_session, indexed_kb, service):
    ticket = make_ticket(db_session, "Приложение вылетает при открытии корзины")
    service.process(db_session, ticket)

    assert ticket.status == TicketStatus.AWAITING_CLARIFICATION
    assert ticket.clarification_count == 1


def test_both_clarification_iterations_are_visible_separately(db_session, indexed_kb, service):
    """UC8: повторная классификация не должна затирать первую."""
    ticket = make_ticket(db_session, "Приложение вылетает при открытии корзины")
    service.process(db_session, ticket)

    db_session.add(
        Message(
            ticket_id=ticket.id,
            sender=MessageSender.CLIENT,
            iteration=1,
            content="Вылетает на Android 14 при открытии корзины, приложение последней версии",
        )
    )
    db_session.flush()
    service.process(db_session, ticket)

    classifications = db_session.scalars(
        select(Classification)
        .where(Classification.ticket_id == ticket.id)
        .order_by(Classification.iteration)
    ).all()

    assert [row.iteration for row in classifications] == [0, 1]
    assert ticket.status == TicketStatus.ESCALATED_STANDARD

    audits = db_session.scalars(
        select(AuditLog).where(AuditLog.ticket_id == ticket.id).order_by(AuditLog.created_at)
    ).all()
    assert [row.rule_id for row in audits] == ["R7a", "R7b"]


def test_client_request_wins_and_gets_priority(db_session, indexed_kb, service):
    ticket = make_ticket(
        db_session,
        "Подскажите, какие способы оплаты доступны?",
        status=TicketStatus.ESCALATED_PRIORITY,
    )
    result = service.process(db_session, ticket)

    escalation = db_session.get(Escalation, result.escalation_id)
    assert result.outcome.decision.rule_id == "R1"
    assert escalation.priority == PRIORITY_CLIENT_REQUESTED
    assert ticket.priority == PRIORITY_CLIENT_REQUESTED
    assert ticket.status == TicketStatus.ESCALATED_PRIORITY


def test_pii_never_reaches_audit_log(db_session, indexed_kb, service):
    """NFR4: оригинал остаётся в messages.content, в трейс идёт статистика."""
    phone = "+7 999 123-45-67"
    ticket = make_ticket(db_session, f"Мой телефон {phone}, где мой заказ?")
    service.process(db_session, ticket)

    audit = db_session.scalar(select(AuditLog).where(AuditLog.ticket_id == ticket.id))
    assert audit.payload["pii_redacted"] == {"phone": 1}
    assert phone not in str(audit.payload)
    assert phone not in (audit.reasoning or "")


def test_ticket_without_client_message_is_rejected(db_session, indexed_kb, service):
    ticket = Ticket(channel="web", external_id="empty-1", status=TicketStatus.NEW.value)
    db_session.add(ticket)
    db_session.flush()

    with pytest.raises(ValueError, match="нет сообщений клиента"):
        service.process(db_session, ticket)
