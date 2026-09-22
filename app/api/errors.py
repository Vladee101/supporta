"""Единая модель ошибок API (раздел «API-контракты»).

    {"error": {"code": "...", "message": "...", "details": ..., "trace_id": "..."}}

`code` - стабильный машиночитаемый идентификатор, на него завязывается
клиент; `message` - для человека и может меняться.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core import tracing


class ApiError(Exception):
    def __init__(
        self, status_code: int, code: str, message: str, details: Any | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _error_response(
    status_code: int, code: str, message: str, details: Any | None = None, headers=None
) -> JSONResponse:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        # Тот же trace_id, что в логах и audit_log запроса (TraceMiddleware).
        "trace_id": tracing.ensure_trace_id(),
    }
    if details is not None:
        body["details"] = jsonable_encoder(details)
    return JSONResponse(status_code=status_code, content={"error": body}, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, exc.details)

    # Ошибки валидации и роутинга FastAPI по умолчанию отдаёт как {"detail": ...} -
    # без кода и trace_id. Модель ошибок одна на весь API, иначе клиенту и консоли
    # пришлось бы разбирать два формата.
    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            422, "validation_error", "запрос не прошёл валидацию", details=exc.errors()
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(
            exc.status_code,
            f"http_{exc.status_code}",
            str(exc.detail),
            headers=getattr(exc, "headers", None),
        )
