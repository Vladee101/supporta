"""Единая модель ошибок API (раздел «API-контракты»).

    {"error": {"code": "...", "message": "...", "details": ..., "trace_id": "..."}}

`code` - стабильный машиночитаемый идентификатор, на него завязывается
клиент; `message` - для человека и может меняться.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(
        self, status_code: int, code: str, message: str, details: Any | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        body: dict[str, Any] = {
            "code": exc.code,
            "message": exc.message,
            "trace_id": request.headers.get("x-trace-id") or uuid.uuid4().hex,
        }
        if exc.details is not None:
            body["details"] = exc.details
        return JSONResponse(status_code=exc.status_code, content={"error": body})
