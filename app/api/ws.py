"""WebSocket-шлюз: push новых эскалаций в Operator Console.

Событие в сокете - **уведомление, а не источник данных**. Консоль по нему
дочитывает контекст REST-ом (`GET /escalations/{id}`), а при обрыве соединения
восстанавливает состояние обычным запросом очереди. Поэтому потерянное или
продублированное уведомление ничего не ломает, и доставку в сокет не нужно
делать надёжной - надёжность обеспечена на уровне outbox и inbox.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.core.auth import InvalidTokenError, verify_token

router = APIRouter()


class ConnectionHub:
    """Подключённые консоли одного экземпляра API."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._clients)

    async def add(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def remove(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, message: dict) -> int:
        """Разослать всем; клиента, на котором отправка упала, отключить."""
        async with self._lock:
            clients = list(self._clients)

        delivered = 0
        for websocket in clients:
            try:
                await websocket.send_json(message)
                delivered += 1
            except Exception:  # noqa: BLE001 - мёртвый сокет не должен ронять рассылку
                await self.remove(websocket)
        return delivered


hub = ConnectionHub()


@router.websocket("/api/v1/ws/escalations")
async def escalations_socket(websocket: WebSocket, token: str | None = Query(default=None)):
    # Браузерный WebSocket не умеет ставить заголовок Authorization,
    # поэтому токен приходит query-параметром.
    try:
        principal = verify_token(token or "")
    except InvalidTokenError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    await hub.add(websocket)
    await websocket.send_json({"type": "hello", "operator_id": str(principal.operator_id)})
    try:
        while True:
            # Входящие сообщения не нужны; чтение держит соединение и ловит закрытие.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.remove(websocket)
        with contextlib.suppress(Exception):
            await websocket.close()
