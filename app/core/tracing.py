"""Сквозной trace_id (раздел «Наблюдаемость»).

Один идентификатор от HTTP-запроса до записи в audit_log, события outbox и
сообщения в RabbitMQ; он же - в теле ошибки и в каждой строке лога. Живёт в
contextvar: FastAPI выполняет синхронные обработчики в пуле потоков с копией
контекста, поэтому значение, выставленное в middleware, видно и там.

Входящий `X-Trace-Id` принимается, только если похож на идентификатор (hex или
UUID): произвольная строка из заголовка попала бы в логи как есть - это
инъекция в логи, а не трассировка. Иначе генерируется новый.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

HEADER = "x-trace-id"
_VALID = re.compile(r"^[0-9a-fA-F-]{8,64}$")

_current: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _current.get()


def ensure_trace_id() -> str:
    """Текущий trace_id или новый - для кода вне HTTP-запроса (воркеры, скрипты)."""
    return _current.get() or new_trace_id()


def set_trace_id(value: str | None) -> str:
    trace_id = value if value and _VALID.match(value) else new_trace_id()
    _current.set(trace_id)
    return trace_id


class TraceMiddleware:
    """ASGI-middleware: trace_id на весь запрос и в заголовке ответа.

    Чистый ASGI, а не BaseHTTPMiddleware: тот выполняет обработчик в отдельной
    задаче, и порядок распространения контекста там неочевиден.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        incoming = None
        for name, value in scope.get("headers", ()):
            if name.decode("latin-1").lower() == HEADER:
                incoming = value.decode("latin-1")
                break
        token = _current.set(None)
        trace_id = set_trace_id(incoming)

        async def send_with_header(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", ()))
                headers.append((HEADER.encode("latin-1"), trace_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            _current.reset(token)
