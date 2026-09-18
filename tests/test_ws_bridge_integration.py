"""Мост RabbitMQ → WebSocket на живом брокере."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pika
import pytest

from app.api.ws import ConnectionHub
from app.core.config import get_settings
from app.messaging.topology import NOTIFY_EXCHANGE
from app.messaging.ws_bridge import NotifyBridge

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def broker_url() -> str:
    url = get_settings().rabbitmq_url
    try:
        pika.BlockingConnection(pika.URLParameters(url)).close()
    except pika.exceptions.AMQPError as exc:
        pytest.skip(f"RabbitMQ недоступен: {exc!r}")
    return url


class _RecordingSocket:
    def __init__(self) -> None:
        self.received: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.received.append(message)


def test_stop_right_after_start_does_not_hang(broker_url):
    """Регрессия: stop() во время подключения вешал поток до таймаута join."""
    loop = asyncio.new_event_loop()
    try:
        bridge = NotifyBridge(broker_url, ConnectionHub(), loop)
        started = time.monotonic()
        bridge.start()
        bridge.stop()
        assert time.monotonic() - started < 2.0
        assert not bridge._thread.is_alive()
    finally:
        loop.close()


def test_notification_reaches_connected_socket(broker_url):
    loop = asyncio.new_event_loop()
    hub = ConnectionHub()
    socket = _RecordingSocket()
    loop.run_until_complete(hub.add(socket))

    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    bridge = NotifyBridge(broker_url, hub, loop)
    bridge.start()
    try:
        time.sleep(1.0)  # мосту нужно объявить свою очередь до публикации
        connection = pika.BlockingConnection(pika.URLParameters(broker_url))
        message = {"type": "escalation.queued", "escalation_id": "bridge-test"}
        connection.channel().basic_publish(NOTIFY_EXCHANGE, "", json.dumps(message).encode())
        connection.close()

        deadline = time.monotonic() + 5
        while not socket.received and time.monotonic() < deadline:
            time.sleep(0.1)
        assert socket.received == [message]
    finally:
        bridge.stop()
        loop.call_soon_threadsafe(loop.stop)
        runner.join(timeout=2)
        loop.close()
