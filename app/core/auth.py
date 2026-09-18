"""Подписанные токены: операторы консоли и клиентский доступ к тикету.

Формат `base64url(payload).base64url(hmac_sha256(payload))`. Это не JWT, но
свойства те же, что нужны здесь: без секрета токен не подделать, у него есть
срок жизни и назначение. Полноценный IdP - вне scope MVP; интерфейс
(`Principal`, `current_operator`) от формата не зависит, замена прозрачна.

Каждый токен несёт тип (`typ`): `operator` или `ticket`. Оба подписаны одним
секретом, и без проверки типа токен тикета, выданный клиенту, открывал бы
операторский API - и наоборот.

WebSocket из браузера не умеет ставить заголовок `Authorization`, поэтому
для WS операторский токен принимается query-параметром - см. `app.api.ws`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass

from fastapi import Depends, Header

from app.api.errors import ApiError
from app.core.config import get_settings
from app.domain.enums import OperatorRole

OPERATOR = "operator"
TICKET = "ticket"


@dataclass(frozen=True, slots=True)
class Principal:
    operator_id: uuid.UUID
    role: OperatorRole


class InvalidTokenError(ValueError):
    pass


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(payload: bytes, secret: str) -> str:
    return _b64encode(hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).digest())


def _issue(claims: dict, *, ttl_seconds: int, secret: str | None, now: float | None) -> str:
    issued = now if now is not None else time.time()
    payload = json.dumps(
        claims | {"exp": int(issued + ttl_seconds)}, separators=(",", ":")
    ).encode("utf-8")
    return f"{_b64encode(payload)}.{_sign(payload, secret or get_settings().auth_secret)}"


def _read(token: str, *, typ: str, secret: str | None, now: float | None) -> dict:
    try:
        encoded_payload, signature = token.split(".")
        payload = _b64decode(encoded_payload)
    except ValueError as exc:
        raise InvalidTokenError("неверный формат токена") from exc

    expected = _sign(payload, secret or get_settings().auth_secret)
    # compare_digest: сравнение за постоянное время, без утечки через тайминг.
    if not hmac.compare_digest(expected, signature):
        raise InvalidTokenError("подпись не совпадает")

    try:
        claims = json.loads(payload)
        expires_at = float(claims["exp"])
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidTokenError("некорректное содержимое токена") from exc

    if claims.get("typ") != typ:
        raise InvalidTokenError("токен выдан для другого назначения")
    if (now if now is not None else time.time()) >= expires_at:
        raise InvalidTokenError("срок действия токена истёк")
    return claims


# --- операторы --------------------------------------------------------------


def issue_token(
    operator_id: uuid.UUID,
    role: OperatorRole,
    *,
    ttl_seconds: int | None = None,
    secret: str | None = None,
    now: float | None = None,
) -> str:
    ttl = ttl_seconds if ttl_seconds is not None else get_settings().operator_token_ttl_hours * 3600
    return _issue(
        {"typ": OPERATOR, "sub": str(operator_id), "role": role.value},
        ttl_seconds=ttl,
        secret=secret,
        now=now,
    )


def verify_token(token: str, *, secret: str | None = None, now: float | None = None) -> Principal:
    claims = _read(token, typ=OPERATOR, secret=secret, now=now)
    try:
        return Principal(operator_id=uuid.UUID(claims["sub"]), role=OperatorRole(claims["role"]))
    except (KeyError, ValueError) as exc:
        raise InvalidTokenError("некорректное содержимое токена") from exc


def current_operator(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ApiError(401, "unauthorized", "требуется заголовок Authorization: Bearer <token>")
    try:
        return verify_token(authorization[7:].strip())
    except InvalidTokenError as exc:
        raise ApiError(401, "unauthorized", str(exc)) from exc


def require_admin(principal: Principal = Depends(current_operator)) -> Principal:  # noqa: B008
    if principal.role is not OperatorRole.ADMIN:
        raise ApiError(403, "forbidden", "операция требует роли admin")
    return principal


# --- клиентский доступ к тикету ----------------------------------------------


def issue_ticket_token(
    ticket_id: uuid.UUID,
    *,
    ttl_seconds: int | None = None,
    secret: str | None = None,
    now: float | None = None,
) -> str:
    """Токен, который клиент получает при создании тикета (NFR4: защита от IDOR)."""
    ttl = ttl_seconds if ttl_seconds is not None else get_settings().ticket_token_ttl_hours * 3600
    return _issue({"typ": TICKET, "sub": str(ticket_id)}, ttl_seconds=ttl, secret=secret, now=now)


def verify_ticket_token(
    token: str, ticket_id: uuid.UUID, *, secret: str | None = None, now: float | None = None
) -> None:
    """Токен должен быть выдан именно на этот тикет - иначе чужой тикет не открыть."""
    claims = _read(token, typ=TICKET, secret=secret, now=now)
    if claims.get("sub") != str(ticket_id):
        raise InvalidTokenError("токен выдан на другой тикет")


def ticket_access(
    ticket_id: uuid.UUID, x_ticket_token: str | None = Header(default=None)
) -> uuid.UUID:
    if not x_ticket_token:
        raise ApiError(401, "unauthorized", "требуется заголовок X-Ticket-Token")
    try:
        verify_ticket_token(x_ticket_token, ticket_id)
    except InvalidTokenError as exc:
        # 404, а не 403: ответ не должен подтверждать, что тикет с таким id существует.
        raise ApiError(404, "not_found", "тикет не найден") from exc
    return ticket_id


# --- подпись вебхуков каналов --------------------------------------------------


def webhook_signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_webhook_signature(body: bytes, header: str | None, secret: str) -> bool:
    return bool(header) and hmac.compare_digest(webhook_signature(body, secret), header)
