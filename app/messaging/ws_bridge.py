"""Мост RabbitMQ → WebSocket внутри процесса API.

Каждый экземпляр API объявляет свою эксклюзивную очередь на fanout-обменнике
`escalations.notify` - так уведомление получает каждый экземпляр, а не один
из них, и каждый рассылает его своим подключённым консолям.

pika синхронный, поэтому мост живёт в отдельном потоке и передаёт сообщения
в event loop FastAPI через `run_coroutine_threadsafe`. Если брокер недоступен,
мост переподключается в фоне, а API продолжает работать: консоль в этом
случае видит новые эскалации при обновлении очереди по REST.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import pika

from app.api.ws import ConnectionHub
from app.messaging.topology import NOTIFY_EXCHANGE, declare_topology

log = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 5.0
#: Как часто поток проверяет флаг остановки - столько же максимум ждёт выход API.
POLL_SECONDS = 0.5


class NotifyBridge:
    def __init__(self, url: str, hub: ConnectionHub, loop: asyncio.AbstractEventLoop) -> None:
        self._url = url
        self._hub = hub
        self._loop = loop
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ws-notify-bridge", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # Поток сам проверяет флаг не реже раза в POLL_SECONDS и закрывает
        # соединение из своего потока. Закрывать его отсюда нельзя надёжно:
        # stop() может прийти, пока поток ещё подключается и соединения нет.
        self._stop.set()
        self._thread.join(timeout=POLL_SECONDS * 5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._consume()
            except pika.exceptions.AMQPError as exc:
                if self._stop.is_set():
                    break
                log.warning("мост уведомлений: брокер недоступен (%r), повтор", exc)
                self._stop.wait(RECONNECT_DELAY_SECONDS)

    def _consume(self) -> None:
        connection = pika.BlockingConnection(pika.URLParameters(self._url))
        try:
            channel = connection.channel()
            declare_topology(channel)
            # Эксклюзивная автоудаляемая очередь: живёт, пока жив этот экземпляр API.
            queue = channel.queue_declare("", exclusive=True, auto_delete=True).method.queue
            channel.queue_bind(queue, NOTIFY_EXCHANGE)

            def on_message(ch, method, _properties, body: bytes) -> None:
                try:
                    message = json.loads(body)
                except ValueError:
                    log.warning("мост уведомлений: некорректное сообщение отброшено")
                else:
                    asyncio.run_coroutine_threadsafe(self._hub.broadcast(message), self._loop)
                ch.basic_ack(method.delivery_tag)

            channel.basic_consume(queue, on_message)
            # Не start_consuming(): он блокирует до закрытия соединения и не видит
            # флаг остановки. Короткие циклы обработки дают выйти за POLL_SECONDS.
            while not self._stop.is_set():
                connection.process_data_events(time_limit=POLL_SECONDS)
        finally:
            if connection.is_open:
                connection.close()
