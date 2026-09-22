"""Применение решения агента к состоянию тикета.

Здесь и только здесь пайплайн пишет в Postgres, и пишет одной транзакцией:
тикет + классификация + найденные документы + эскалация + `outbox_event`
(ADR-004, ADR-007). Если публикация в RabbitMQ невозможна - это проблема
поллера, а не согласованности: событие уже лежит в той же транзакции, что и
эскалация, и не может потеряться между commit и ack брокеру.

Каждый прогон пишет строку в `audit_log` с `rule_id`, обоими confidence и
идентификаторами моделей - это и есть трассируемость из NFR3.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import metrics
from app.agent.graph import AgentOutcome, TicketGraph
from app.core import tracing
from app.db.base import release_connection
from app.db.models import (
    AuditLog,
    Classification,
    Message,
    RagRetrieval,
    Ticket,
)
from app.domain.enums import (
    Action,
    MessageSender,
    TicketStatus,
)
from app.escalations.writer import create_escalation

#: Статус тикета по действию. R-default попадает сюда как обычная эскалация -
#: клиент не должен страдать от дефекта таблицы, но алерт уйдёт (см. audit_log).
_STATUS_BY_ACTION = {
    Action.AUTO_ANSWER: TicketStatus.RESOLVED_AUTO,
    Action.CLARIFY: TicketStatus.AWAITING_CLARIFICATION,
    Action.ESCALATE: TicketStatus.ESCALATED_STANDARD,
    Action.PRIORITY_ESCALATE: TicketStatus.ESCALATED_PRIORITY,
}


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    ticket_id: uuid.UUID
    trace_id: str
    outcome: AgentOutcome
    escalation_id: uuid.UUID | None


class AgentService:
    def __init__(self, graph: TicketGraph) -> None:
        self._graph = graph

    def process(self, session: Session, ticket: Ticket) -> ProcessingResult:
        started = time.monotonic()
        text = self._latest_client_message(session, ticket)
        # Тот же trace_id, что у HTTP-запроса (раздел «Наблюдаемость»).
        trace_id = tracing.ensure_trace_id()
        # Дальше - классификация через LLM: соединение с базой на это время не нужно.
        release_connection(session)

        outcome = self._graph.run(
            session,
            text,
            iteration=ticket.clarification_count,
            human_requested=ticket.status == TicketStatus.ESCALATED_PRIORITY,
        )

        iteration = ticket.clarification_count
        escalation_id = self._persist(session, ticket, outcome, iteration, trace_id)
        session.commit()
        # После commit'а: откатившийся проход не должен попасть в SLI.
        metrics.observe_ticket(outcome, time.monotonic() - started)

        return ProcessingResult(
            ticket_id=ticket.id,
            trace_id=trace_id,
            outcome=outcome,
            escalation_id=escalation_id,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _latest_client_message(session: Session, ticket: Ticket) -> str:
        message = session.scalar(
            select(Message)
            .where(Message.ticket_id == ticket.id, Message.sender == MessageSender.CLIENT)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        if message is None:
            raise ValueError(f"у тикета {ticket.id} нет сообщений клиента")
        return message.content

    def _persist(
        self,
        session: Session,
        ticket: Ticket,
        outcome: AgentOutcome,
        iteration: int,
        trace_id: str,
    ) -> uuid.UUID | None:
        now = datetime.now(UTC)
        decision = outcome.decision

        if outcome.classification is not None:
            classification = outcome.classification
            session.add(
                Classification(
                    ticket_id=ticket.id,
                    iteration=iteration,
                    category=classification.category.value,
                    risk_level=classification.risk_level.value,
                    confidence=classification.confidence,
                    confidence_source=(
                        classification.confidence_source.value
                        if classification.confidence_source
                        else None
                    ),
                    model_id=classification.model_id,
                    reasoning=classification.reasoning,
                )
            )
            ticket.category = classification.category.value
            ticket.class_confidence = classification.confidence
            ticket.risk_level = classification.risk_level.value

        if outcome.retrieval is not None:
            for chunk in outcome.retrieval.chunks:
                session.add(
                    RagRetrieval(
                        ticket_id=ticket.id,
                        document_version_id=chunk.document_version_id,
                        iteration=iteration,
                        rank=chunk.rank,
                        relevance_score=chunk.relevance_score,
                        # Снапшот: чанкинг и текст документа могут измениться,
                        # а трейс должен показывать использованное (ADR-011).
                        chunk_snapshot=chunk.content,
                    )
                )

        ticket.status = _STATUS_BY_ACTION[decision.action].value
        ticket.priority = max(ticket.priority, decision.priority)

        if outcome.reply_text:
            session.add(
                Message(
                    ticket_id=ticket.id,
                    sender=MessageSender.AGENT,
                    iteration=iteration,
                    content=outcome.reply_text,
                    content_redacted=outcome.reply_text,
                )
            )

        if decision.action is Action.CLARIFY:
            ticket.clarification_count = iteration + 1
        if decision.action is Action.AUTO_ANSWER:
            ticket.resolved_at = now

        escalation_id: uuid.UUID | None = None
        if decision.is_escalation:
            escalation_id = self._create_escalation(session, ticket, outcome, trace_id)

        session.add(
            AuditLog(
                ticket_id=ticket.id,
                actor="agent",
                action=decision.action.value,
                rule_id=decision.rule_id,
                class_confidence=(
                    outcome.classification.confidence if outcome.classification else None
                ),
                rag_confidence=(outcome.retrieval.rag_confidence if outcome.retrieval else None),
                payload=self._audit_payload(outcome, iteration),
                reasoning=(outcome.classification.reasoning if outcome.classification else None),
                trace_id=trace_id,
            )
        )
        return escalation_id

    def _create_escalation(
        self, session: Session, ticket: Ticket, outcome: AgentOutcome, trace_id: str
    ) -> uuid.UUID:
        decision = outcome.decision
        # Инвариант Decision Engine: у любой эскалации есть причина
        # (проверяется тестом test_escalations_carry_reason_and_answers_do_not).
        assert decision.reason is not None, f"эскалация без причины: {decision.rule_id}"
        escalation = create_escalation(
            session,
            ticket,
            reason=decision.reason,
            rule_id=decision.rule_id,
            priority=decision.priority,
            category=outcome.category.value,
            draft_text=outcome.draft.text if outcome.draft else None,
            trace_id=trace_id,
        )
        return escalation.id

    @staticmethod
    def _audit_payload(outcome: AgentOutcome, iteration: int) -> dict:
        retrieval = outcome.retrieval
        return {
            "iteration": iteration,
            "category": outcome.category.value,
            "rule_id": outcome.decision.rule_id,
            "alert": outcome.decision.alert,
            "pii_redacted": outcome.redaction.counts,
            "classifier_model": (
                outcome.classification.model_id if outcome.classification else None
            ),
            "confidence_source": (
                outcome.classification.confidence_source.value
                if outcome.classification and outcome.classification.confidence_source
                else None
            ),
            "generator_model": outcome.draft.model_id if outcome.draft else None,
            # SLI «стоимость на тикет» (NFR5): расход считается по факту ответов
            # провайдера, а не оценкой по прайсу.
            "llm_usage": outcome.llm_usage,
            "sources": list(outcome.draft.sources) if outcome.draft else [],
            "retrieved": [
                {
                    "slug": chunk.slug,
                    "rank": chunk.rank,
                    "score": chunk.relevance_score,
                    "version_id": str(chunk.document_version_id),
                }
                for chunk in (retrieval.chunks if retrieval else ())
            ],
        }
