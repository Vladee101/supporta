"""Эндпоинты Operator Console для эскалаций (раздел «API-контракты»)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import metrics
from app.api.errors import ApiError
from app.core.auth import Principal, current_operator
from app.core.config import get_settings
from app.db.base import get_session
from app.domain.enums import OperatorActionType
from app.escalations import operations as ops

router = APIRouter(prefix="/api/v1/escalations", tags=["escalations"])


class ResolveRequest(BaseModel):
    action: OperatorActionType
    final_text: str | None = Field(default=None, max_length=10_000)


def _now() -> datetime:
    return datetime.now(UTC)


def _translate(exc: ops.EscalationError) -> ApiError:
    return ApiError(exc.status_code, exc.code, str(exc))


@router.get("")
def list_queue(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    session: Session = Depends(get_session),  # noqa: B008
    _: Principal = Depends(current_operator),  # noqa: B008
) -> dict:
    try:
        page = ops.queue(session, now=_now(), limit=limit, cursor=cursor)
    except ops.EscalationError as exc:
        raise _translate(exc) from exc
    return {
        "items": [ops.serialize_escalation(item) for item in page.items],
        "next_cursor": page.next_cursor,
    }


@router.get("/{escalation_id}")
def get_context(
    escalation_id: uuid.UUID,
    session: Session = Depends(get_session),  # noqa: B008
    _: Principal = Depends(current_operator),  # noqa: B008
) -> dict:
    try:
        return ops.context(session, escalation_id)
    except ops.EscalationError as exc:
        raise _translate(exc) from exc


@router.post("/{escalation_id}/claim")
def claim(
    escalation_id: uuid.UUID,
    session: Session = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(current_operator),  # noqa: B008
) -> dict:
    ttl = timedelta(minutes=get_settings().escalation_claim_ttl_minutes)
    try:
        escalation = ops.claim(session, escalation_id, principal.operator_id, now=_now(), ttl=ttl)
    except ops.EscalationError as exc:
        raise _translate(exc) from exc
    return ops.serialize_escalation(escalation)


@router.delete("/{escalation_id}/claim")
def release(
    escalation_id: uuid.UUID,
    session: Session = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(current_operator),  # noqa: B008
) -> dict:
    try:
        escalation = ops.release(session, escalation_id, principal.operator_id, now=_now())
    except ops.EscalationError as exc:
        raise _translate(exc) from exc
    return ops.serialize_escalation(escalation)


@router.post("/{escalation_id}/resolve")
def resolve(
    escalation_id: uuid.UUID,
    body: ResolveRequest,
    session: Session = Depends(get_session),  # noqa: B008
    principal: Principal = Depends(current_operator),  # noqa: B008
) -> dict:
    try:
        action = ops.resolve(
            session,
            escalation_id,
            principal.operator_id,
            action=body.action,
            final_text=body.final_text,
            now=_now(),
        )
    except ops.EscalationError as exc:
        raise _translate(exc) from exc
    # SLI «доля правок оператора» (раздел «Наблюдаемость»); resolve уже закоммичен.
    metrics.observe_operator_action(action.action_type)
    return {
        "escalation_id": str(escalation_id),
        "action": action.action_type,
        "final_text": action.final_text,
    }
