"""Топология RabbitMQ (ADR-003).

    outbox poller ──► [escalations] topic ──escalation.#──► escalations.created
                                                            (x-max-priority, DLX)
                                                                   │ nack
                                                                   ▼
                                               [escalations.dlx] ──► escalations.dead

    escalation consumer ──► [escalations.notify] fanout ──► эксклюзивная очередь
                                                            каждого экземпляра API

Две ветки намеренно разные. Рабочая очередь - одна на всех consumer'ов:
эскалацию обрабатывает ровно один из них. Уведомления - fanout: каждому
экземпляру API нужна своя копия, чтобы отправить её своим WebSocket-клиентам.

Объявления идемпотентны: любой процесс может вызвать `declare_topology` при
старте, повторное объявление с теми же аргументами - no-op.
"""

from __future__ import annotations

from pika.adapters.blocking_connection import BlockingChannel

EXCHANGE = "escalations"
QUEUE = "escalations.created"
ROUTING_KEY = "escalation.created"
BINDING_KEY = "escalation.#"

DEAD_LETTER_EXCHANGE = "escalations.dlx"
DEAD_LETTER_QUEUE = "escalations.dead"

NOTIFY_EXCHANGE = "escalations.notify"

#: Приоритет клиентского запроса (A4) - 10, стандартной эскалации - 0.
#: RabbitMQ поддерживает до 255 уровней, но каждый уровень стоит памяти,
#: а нам нужны два - берём ровно столько, сколько есть в домене.
MAX_PRIORITY = 10


def declare_topology(channel: BlockingChannel) -> None:
    channel.exchange_declare(DEAD_LETTER_EXCHANGE, exchange_type="fanout", durable=True)
    channel.queue_declare(DEAD_LETTER_QUEUE, durable=True)
    channel.queue_bind(DEAD_LETTER_QUEUE, DEAD_LETTER_EXCHANGE)

    channel.exchange_declare(EXCHANGE, exchange_type="topic", durable=True)
    channel.queue_declare(
        QUEUE,
        durable=True,
        arguments={
            "x-max-priority": MAX_PRIORITY,
            "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
        },
    )
    channel.queue_bind(QUEUE, EXCHANGE, routing_key=BINDING_KEY)

    channel.exchange_declare(NOTIFY_EXCHANGE, exchange_type="fanout", durable=True)
