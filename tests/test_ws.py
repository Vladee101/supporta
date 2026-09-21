"""WebSocket-шлюз: аутентификация и рассылка уведомлений."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.ws import ConnectionHub, hub
from app.core.auth import issue_token
from app.domain.enums import OperatorRole
from app.main import app


def test_socket_rejects_missing_or_bad_token():
    urls = ("/api/v1/ws/escalations", "/api/v1/ws/escalations?token=forged")
    with TestClient(app) as client:
        for url in urls:
            with (
                pytest.raises(WebSocketDisconnect) as excinfo,
                client.websocket_connect(url) as socket,
            ):
                socket.receive_json()
            assert excinfo.value.code == 1008


def test_socket_receives_broadcast():
    operator_id = uuid.uuid4()
    token = issue_token(operator_id, OperatorRole.OPERATOR)

    with (
        TestClient(app) as client,
        client.websocket_connect(f"/api/v1/ws/escalations?token={token}") as socket,
    ):
        assert socket.receive_json() == {"type": "hello", "operator_id": str(operator_id)}

        message = {"type": "escalation.queued", "escalation_id": "e1", "priority": 10}
        delivered = client.portal.call(hub.broadcast, message)

        assert delivered == 1
        assert socket.receive_json() == message


class _DeadSocket:
    async def send_json(self, message):
        raise RuntimeError("connection reset")


class _LiveSocket:
    def __init__(self):
        self.received = []

    async def send_json(self, message):
        self.received.append(message)


def test_dead_client_is_dropped_without_breaking_broadcast():
    async def scenario():
        local_hub = ConnectionHub()
        dead, live = _DeadSocket(), _LiveSocket()
        await local_hub.add(dead)
        await local_hub.add(live)

        delivered = await local_hub.broadcast({"type": "ping"})
        return delivered, local_hub.size, live.received

    delivered, size, received = asyncio.run(scenario())
    assert delivered == 1
    assert size == 1
    assert received == [{"type": "ping"}]
