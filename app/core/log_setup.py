"""Структурные логи с PII-редакцией (раздел «Наблюдаемость», NFR4).

Каждая строка - JSON-объект с `trace_id`, поэтому логи API и воркеров
связываются с audit_log и событиями RabbitMQ по одному идентификатору.

Редакция применяется к итоговому тексту записи целиком, включая трейсбек:
в тексте исключения легко оказывается текст обращения из аргументов функции,
и редакция одного лишь сообщения его бы пропустила. Это тот же редактор, что
стоит перед вызовами LLM, - другого определения PII в системе нет.

Формат `text` - для локальной разработки: читать глазами удобнее, редакция
та же.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

from app.core.tracing import current_trace_id
from app.services.pii import redact

LogFormat = Literal["json", "text"]


class TraceFilter(logging.Filter):
    """Добавляет trace_id текущего запроса (или воркерной операции) в запись."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "trace_id", None):
            record.trace_id = current_trace_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()).text,
            "trace_id": getattr(record, "trace_id", None),
        }
        if record.exc_info:
            entry["exception"] = redact(self.formatException(record.exc_info)).text
        return json.dumps(entry, ensure_ascii=False)


class RedactingTextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record)).text


def configure_logging(fmt: LogFormat = "json", level: int = logging.INFO) -> None:
    """Один обработчик на корневом логгере; uvicorn пишет через него же."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(TraceFilter())
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else RedactingTextFormatter("%(asctime)s %(name)s %(levelname)s [%(trace_id)s] %(message)s")
    )
    handler._support_log_handler = True  # type: ignore[attr-defined]
    root = logging.getLogger()
    # Заменяется только свой обработчик: повторный вызов (второй lifespan,
    # тесты) не должен ни дублировать строки, ни снимать чужие обработчики.
    root.handlers = [h for h in root.handlers if not getattr(h, "_support_log_handler", False)] + [
        handler
    ]
    root.setLevel(level)
    # У uvicorn свои обработчики в обход корневого - без этого его строки
    # (в том числе access-лог с путями запросов) шли бы мимо редакции и JSON.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # Библиотеки пишут INFO на каждое соединение: pika - около десятка строк на
    # каждый опрос /metrics (раз в 15 с), httpx - строку на каждый вызов LLM.
    # Это шум, в котором тонут строки, по которым разбирают инциденты.
    for name in ("pika", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
