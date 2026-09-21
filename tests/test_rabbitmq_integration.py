"""Проверка на живом RabbitMQ: топология, приоритеты, confirm'ы, DLX.

Тесты работают на отдельном виртуальном окружении имён - с префиксом
`test.` у обменников и очередей - и удаляют их после себя, чтобы не мешать
рабочим очередям разработчика. Если брокер недоступен, тесты пропускаются
(в CI с `REQUIRE_INTEGRATION=1` - падают).
"""

from __future__ import annotations

import json
import time
import uuid

import pika
import pytest

from app.core.config import get_settings
from app.messaging.publisher import PikaPublisher, PublishError
from tests.conftest import skip_unless_required

pytestmark = pytest.mark.integration

PREFIX = f"test.{uuid.uuid4().hex[:8]}"
EXCHANGE = f"{PREFIX}.escalations"
QUEUE = f"{PREFIX}.escalations.created"
DLX = f"{PREFIX}.escalations.dlx"
DEAD = f"{PREFIX}.escalations.dead"


@pytest.fixture(scope="module")
def channel():
    try:
        connection = pika.BlockingConnection(pika.URLParameters(get_settings().rabbitmq_url))
    except pika.exceptions.AMQPError as exc:
        skip_unless_required(f"RabbitMQ недоступен: {exc!r}")

    ch = connection.channel()
    # Та же форма топологии, что в app.messaging.topology, но с тестовыми именами.
    # Очереди durable: RabbitMQ 4 запрещает transient-очереди без exclusive,
    # поэтому временность обеспечивается удалением в teardown.
    ch.exchange_declare(DLX, exchange_type="fanout", auto_delete=True)
    ch.queue_declare(DEAD, durable=True)
    ch.queue_bind(DEAD, DLX)
    ch.exchange_declare(EXCHANGE, exchange_type="topic", auto_delete=True)
    ch.queue_declare(
        QUEUE,
        durable=True,
        arguments={"x-max-priority": 10, "x-dead-letter-exchange": DLX},
    )
    ch.queue_bind(QUEUE, EXCHANGE, routing_key="escalation.#")

    yield ch

    for queue in (QUEUE, DEAD):
        ch.queue_delete(queue)
    ch.exchange_delete(EXCHANGE)
    ch.exchange_delete(DLX)
    connection.close()


@pytest.fixture
def publisher():
    instance = PikaPublisher(get_settings().rabbitmq_url, exchange=EXCHANGE)
    yield instance
    instance.close()


def _drain(ch, queue):
    messages = []
    while True:
        method, properties, body = ch.basic_get(queue, auto_ack=True)
        if method is None:
            return messages
        messages.append((properties, json.loads(body)))


def _wait_for(ch, queue, count, timeout=5.0):
    """Дождаться `count` сообщений в очереди.

    Dead-lettering асинхронный: `basic_nack` не ждёт ответа брокера, и в момент
    чтения DLQ сообщение может быть ещё в пути. Одиночная проверка очереди
    сразу после nack давала гонку - на быстром раннере CI очередь была пуста.
    """
    messages = []
    deadline = time.monotonic() + timeout
    while len(messages) < count and time.monotonic() < deadline:
        messages.extend(_drain(ch, queue))
        if len(messages) < count:
            time.sleep(0.05)
    return messages


def test_priority_message_overtakes_standard_ones(channel, publisher):
    """Приоритетная очередь: клиентский запрос (A4) обгоняет стандартные эскалации."""
    for index in range(3):
        publisher.publish(
            "escalation.created",
            json.dumps({"n": index}).encode(),
            message_id=f"std-{index}",
            priority=0,
        )
    publisher.publish(
        "escalation.created", json.dumps({"n": "vip"}).encode(), message_id="vip", priority=10
    )

    received = _drain(channel, QUEUE)
    assert [body["n"] for _, body in received] == ["vip", 0, 1, 2]
    assert received[0][0].message_id == "vip"
    assert received[0][0].delivery_mode == 2  # persistent


def test_dead_connection_is_replaced_and_publish_retried(channel, publisher):
    """Регрессия: после простоя брокер закрывал соединение поллера, и первая
    публикация падала, откладывая эскалацию на период бэкоффа."""

    class DeadChannel:
        is_closed = False  # клиент ещё считает канал живым

        def basic_publish(self, **_):
            raise pika.exceptions.StreamLostError("connection reset by broker")

    publisher._channel = DeadChannel()
    publisher.publish("escalation.created", json.dumps({"n": "retry"}).encode(), message_id="retry")

    assert [body["n"] for _, body in _drain(channel, QUEUE)] == ["retry"]


def test_unroutable_message_is_an_error_not_a_silent_loss(publisher):
    """mandatory + confirm: сообщение, которое некуда положить, - исключение."""
    with pytest.raises(PublishError):
        publisher.publish("nobody.listens", b"{}", message_id="lost", priority=0)


def test_rejected_message_goes_to_dead_letter_queue(channel, publisher):
    publisher.publish("escalation.created", b'{"bad": true}', message_id="poison", priority=0)

    method, _, _ = channel.basic_get(QUEUE, auto_ack=False)
    channel.basic_nack(method.delivery_tag, requeue=False)

    dead = _wait_for(channel, DEAD, count=1)
    assert [properties.message_id for properties, _ in dead] == ["poison"]
