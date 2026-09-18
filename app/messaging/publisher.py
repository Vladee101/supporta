"""Публикация событий в RabbitMQ.

Публикация считается состоявшейся только после publisher confirm: пока брокер
не подтвердил приём, событие в outbox не помечается опубликованным. Без
confirm'ов «успешный» `basic_publish` означал бы лишь «байты ушли в сокет».
"""

from __future__ import annotations

import contextlib
from typing import Protocol

import pika
from pika.adapters.blocking_connection import BlockingChannel

from app.messaging.topology import EXCHANGE, MAX_PRIORITY, declare_topology


class PublishError(RuntimeError):
    """Брокер не подтвердил приём сообщения."""


class EventPublisher(Protocol):
    def publish(
        self, routing_key: str, body: bytes, *, message_id: str, priority: int = 0
    ) -> None:
        """Вернуть управление только после подтверждения брокера, иначе - исключение."""
        ...


class PikaPublisher:
    def __init__(self, url: str, exchange: str = EXCHANGE) -> None:
        self._url = url
        self._exchange = exchange
        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None

    def _ensure_channel(self) -> BlockingChannel:
        if self._channel is None or self._channel.is_closed:
            self._connection = pika.BlockingConnection(pika.URLParameters(self._url))
            self._channel = self._connection.channel()
            declare_topology(self._channel)
            self._channel.confirm_delivery()
        return self._channel

    def publish(
        self, routing_key: str, body: bytes, *, message_id: str, priority: int = 0
    ) -> None:
        properties = pika.BasicProperties(
            content_type="application/json",
            delivery_mode=pika.DeliveryMode.Persistent,
            message_id=message_id,
            priority=max(0, min(priority, MAX_PRIORITY)),
        )
        # Две попытки: первая может упереться в соединение, которое брокер уже
        # закрыл, а клиент ещё считает открытым. Вторая идёт по свежему.
        for attempt in (1, 2):
            channel = self._ensure_channel()
            try:
                channel.basic_publish(
                    exchange=self._exchange,
                    routing_key=routing_key,
                    body=body,
                    properties=properties,
                    # mandatory: сообщение, которое некуда маршрутизировать, - ошибка,
                    # а не тихая потеря.
                    mandatory=True,
                )
                return
            except (pika.exceptions.UnroutableError, pika.exceptions.NackError) as exc:
                raise PublishError(f"брокер отклонил сообщение {message_id}: {exc}") from exc
            except pika.exceptions.AMQPError as exc:
                self.close()
                if attempt == 2:
                    raise PublishError(f"сбой соединения с брокером: {exc!r}") from exc

    def keepalive(self) -> None:
        """Обслужить heartbeat'ы в простое.

        BlockingConnection отвечает на heartbeat'ы брокера только когда клиент
        обращается к соединению. Поллер, у которого долго нет событий, иначе
        молчит дольше таймаута heartbeat - брокер закрывает соединение, и первая
        же публикация после паузы падает.
        """
        if self._connection is None or not self._connection.is_open:
            return
        try:
            self._connection.process_data_events(time_limit=0)
        except pika.exceptions.AMQPError:
            self.close()

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._channel = None
        if connection is not None and connection.is_open:
            # Соединение может быть уже мертво - тогда закрывать нечего.
            with contextlib.suppress(pika.exceptions.AMQPError):
                connection.close()
