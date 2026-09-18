"""Клиентский канал: приём тикетов, сообщения, запрос оператора (UC1, UC4, UC9).

Приём защищён подписью вебхука канала, последующие операции - токеном тикета,
выданным при создании. Пользовательских сессий у клиента нет.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.agent.factory import get_agent_service
from app.agent.service import AgentService
from app.api.errors import ApiError
from app.core.auth import issue_ticket_token, ticket_access, verify_webhook_signature
from app.core.config import get_settings
from app.db.base import get_session
from app.domain.decision import PRIORITY_STANDARD
from app.domain.enums import EscalationReason, TicketStatus
from app.escalations.writer import create_escalation
from app.services.llm import LLMUnavailableError
from app.tickets import service as tickets

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])

MAX_MESSAGE_LENGTH = 5_000


class TicketIn(BaseModel):
    channel: str = Field(min_length=1, max_length=32)
    external_id: str = Field(min_length=1, max_length=128)
    client_id: str | None = Field(default=None, max_length=128)
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    metadata: dict = Field(default_factory=dict)


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


def _translate(exc: tickets.TicketError) -> ApiError:
    return ApiError(exc.status_code, exc.code, str(exc))


def _run_agent(session: Session, agent: AgentService, ticket) -> str | None:
    """Прогнать агента. Недоступность LLM ведёт к эскалации, а не к ошибке (NFR6)."""
    try:
        result = agent.process(session, ticket)
    except LLMUnavailableError:
        session.rollback()
        log.warning("LLM недоступен, тикет %s эскалирован", ticket.id)
        create_escalation(
            session,
            ticket,
            reason=EscalationReason.LLM_UNAVAILABLE,
            rule_id="NFR6",
            priority=PRIORITY_STANDARD,
            category=ticket.category or "unclassified",
        )
        ticket.status = TicketStatus.ESCALATED_STANDARD.value
        session.commit()
        return None
    return result.outcome.reply_text


@router.post("", status_code=201)
async def create_ticket(
    request: Request,
    response: Response,
    x_signature: str | None = Header(default=None),
    session: Session = Depends(get_session),  # noqa: B008
    agent: AgentService = Depends(get_agent_service),  # noqa: B008
) -> dict:
    body = await request.body()
    try:
        payload = TicketIn.model_validate(json.loads(body or b"{}"))
    except (ValueError, ValidationError) as exc:
        raise ApiError(400, "invalid_payload", "тело запроса не прошло валидацию") from exc

    # Порядок проверок - из AC UC1: неподдерживаемый канал отклоняется до всего остального.
    secret = get_settings().channel_secrets.get(payload.channel)
    if secret is None:
        raise ApiError(422, "unsupported_channel", f"канал {payload.channel!r} не поддерживается")
    if not verify_webhook_signature(body, x_signature, secret):
        raise ApiError(401, "invalid_signature", "подпись вебхука не совпадает")

    # Эндпоинт асинхронный только ради сырого тела (подпись считается по байтам).
    # БД и агент синхронные - их вызов прямо здесь заблокировал бы event loop
    # и вместе с ним все остальные запросы, включая WebSocket консоли.
    result = await run_in_threadpool(_intake, session, agent, payload)
    if result["duplicate"]:
        response.status_code = 200
    return result


def _intake(session: Session, agent: AgentService, payload: TicketIn) -> dict:
    intake = tickets.create_ticket(
        session,
        channel=payload.channel,
        external_id=payload.external_id,
        content=payload.content,
        client_id=payload.client_id,
    )
    ticket = intake.ticket
    # Повторная доставка вебхука: тот же тикет, без повторного прогона агента.
    reply = _run_agent(session, agent, ticket) if intake.created else None
    return {
        "ticket_id": str(ticket.id),
        "ticket_token": issue_ticket_token(ticket.id),
        "status": ticket.status,
        "reply": reply,
        "duplicate": not intake.created,
    }


@router.get("/{ticket_id}")
def get_ticket(
    ticket_id: uuid.UUID = Depends(ticket_access),  # noqa: B008
    session: Session = Depends(get_session),  # noqa: B008
) -> dict:
    try:
        return tickets.public_view(session, tickets.get_ticket(session, ticket_id))
    except tickets.TicketError as exc:
        raise _translate(exc) from exc


@router.post("/{ticket_id}/messages")
def add_message(
    body: MessageIn,
    ticket_id: uuid.UUID = Depends(ticket_access),  # noqa: B008
    session: Session = Depends(get_session),  # noqa: B008
    agent: AgentService = Depends(get_agent_service),  # noqa: B008
) -> dict:
    try:
        ticket = tickets.get_ticket(session, ticket_id)
        run_agent = tickets.add_client_message(
            session, ticket, body.content, now=datetime.now(UTC)
        )
    except tickets.TicketError as exc:
        raise _translate(exc) from exc

    reply = _run_agent(session, agent, ticket) if run_agent else None
    return {"ticket_id": str(ticket.id), "status": ticket.status, "reply": reply}


@router.post("/{ticket_id}/escalate")
def escalate(
    ticket_id: uuid.UUID = Depends(ticket_access),  # noqa: B008
    session: Session = Depends(get_session),  # noqa: B008
) -> dict:
    window = timedelta(hours=get_settings().client_escalation_window_hours)
    try:
        ticket = tickets.get_ticket(session, ticket_id)
        escalation = tickets.request_human(session, ticket, now=datetime.now(UTC), window=window)
    except tickets.TicketError as exc:
        raise _translate(exc) from exc

    return {
        "ticket_id": str(ticket.id),
        "status": ticket.status,
        "escalation_id": str(escalation.id),
        "priority": escalation.priority,
    }
