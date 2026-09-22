"""FastAPI-приложение.

Слой агента (классификация → RAG → генерация) вызывается сервисным слоем
внутри процесса, не через сеть. Эскалации приходят в консоль двумя путями:
REST (очередь, контекст, claim, resolve) и WebSocket-уведомления через мост
из RabbitMQ; второй путь опционален, первый - источник истины.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from anyio import to_thread
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from app.api import audit, escalations, kb, tickets, ws
from app.api.errors import install_error_handlers
from app.core.config import get_settings, thresholds
from app.core.log_setup import configure_logging
from app.core.singleton import once
from app.core.tracing import TraceMiddleware
from app.db.base import get_session_factory
from app.domain.decision import RULES
from app.metrics import StateCollector


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_format)
    # Синхронные обработчики (агент, БД) работают в пуле потоков anyio. По
    # умолчанию в нём 40 потоков, а тикет занимает поток на всё время ожидания
    # LLM - при 50 одновременных тикетах (NFR8) часть из них ждала бы в очереди
    # целый цикл обработки, что и показал нагрузочный тест.
    to_thread.current_default_thread_limiter().total_tokens = settings.api_worker_threads
    bridge = None
    if settings.ws_bridge_enabled:
        from app.messaging.ws_bridge import NotifyBridge

        bridge = NotifyBridge(settings.rabbitmq_url, ws.hub, asyncio.get_running_loop())
        bridge.start()
    try:
        yield
    finally:
        if bridge is not None:
            bridge.stop()


app = FastAPI(title="Support Agent", version="0.1.0", lifespan=lifespan)
install_error_handlers(app)
# Сквозной trace_id: заголовок X-Trace-Id на входе и выходе, контекст на весь запрос.
app.add_middleware(TraceMiddleware)
app.include_router(tickets.router)
app.include_router(audit.router)
app.include_router(escalations.router)
app.include_router(kb.router)
app.include_router(ws.router)


@once
def _register_state_collector() -> None:
    # Один раз на процесс: повторная регистрация в REGISTRY - ошибка.
    REGISTRY.register(StateCollector(get_session_factory(), get_settings().rabbitmq_url))


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    """SLI для Prometheus (раздел «Наблюдаемость»).

    Без аутентификации, как принято для /metrics: эндпоинт должен быть доступен
    только из внутренней сети - в метриках нет PII, но есть объёмы и стоимость.
    """
    _register_state_collector()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


#: Сборка Operator Console (`npm run build` в console/). Если её нет - API
#: работает без UI, консоль можно поднять dev-сервером Vite.
CONSOLE_DIST = Path(__file__).resolve().parent.parent / "console" / "dist"
if CONSOLE_DIST.is_dir():
    app.mount("/console", StaticFiles(directory=CONSOLE_DIST, html=True), name="console")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/console/")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/_internal/decision-table")
def decision_table() -> dict[str, object]:
    """Действующая таблица маршрутизации и пороги.

    Диагностический эндпоинт: при разборе инцидента видно, какие пороги
    реально стоят в конфиге - «пороги в конфиге, а не в коде» проверяемо.
    """
    t = thresholds()
    return {
        "thresholds": {
            "class_confidence": t.class_confidence,
            "rag_confidence": t.rag_confidence,
            "max_clarifications": t.max_clarifications,
        },
        "embedding_model": get_settings().embedding_model,
        "rules": [
            {
                "rule_id": rule.rule_id,
                "action": rule.action.value,
                "reason": rule.reason.value if rule.reason else None,
                "description": rule.description,
            }
            for rule in RULES
        ],
    }
