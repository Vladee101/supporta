"""Audit trail решения агента по тикету (UC8, NFR3).

Хронология собирается из append-only `audit_log`, плюс все итерации
классификации отдельно - AC UC8 требует видеть их раздельно, с пометкой о
сработавшем лимите уточнений. Запрос идёт по индексу
`ix_audit_log_ticket_created_at` - целевое время ответа ≤ 2 сек (NFR3).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.api.errors import ApiError
from app.core.auth import Principal, current_operator
from app.db.base import get_session
from app.db.models import AuditLog, Classification, Ticket

router = APIRouter(prefix="/api/v1/tickets", tags=["audit"])


@router.get("/{ticket_id}/audit")
def audit_trail(
    ticket_id: uuid.UUID,
    session=Depends(get_session),  # noqa: B008
    _: Principal = Depends(current_operator),  # noqa: B008
) -> dict:
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise ApiError(404, "not_found", "тикет не найден")

    events = session.scalars(
        select(AuditLog).where(AuditLog.ticket_id == ticket_id).order_by(AuditLog.created_at)
    ).all()
    classifications = session.scalars(
        select(Classification)
        .where(Classification.ticket_id == ticket_id)
        .order_by(Classification.iteration)
    ).all()

    return {
        "ticket": {
            "id": str(ticket.id),
            "status": ticket.status,
            "category": ticket.category,
            "clarification_count": ticket.clarification_count,
        },
        "clarification_limit_reached": any(event.rule_id == "R7b" for event in events),
        "classifications": [
            {
                "iteration": row.iteration,
                "category": row.category,
                "confidence": row.confidence,
                "confidence_source": row.confidence_source,
                "model_id": row.model_id,
                "reasoning": row.reasoning,
                "created_at": row.created_at.isoformat(),
            }
            for row in classifications
        ],
        "events": [
            {
                "at": event.created_at.isoformat(),
                "actor": event.actor,
                "action": event.action,
                "rule_id": event.rule_id,
                "class_confidence": event.class_confidence,
                "rag_confidence": event.rag_confidence,
                "reasoning": event.reasoning,
                "payload": event.payload,
                "trace_id": event.trace_id,
            }
            for event in events
        ],
    }
