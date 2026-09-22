"""Процесс consumer'а эскалаций.

    python -m app.workers.escalation_consumer

Разбор отказов:

* `PoisonMessageError` (нет message_id, битый payload, эскалации нет в БД) -
  nack без requeue: сообщение уходит в DLX `escalations.dead` и ждёт разбора
  по runbook. Повторная доставка его не исправит;
* `OperationalError` (Postgres недоступен) - nack с requeue после паузы:
  сбой временный, сообщение должно быть обработано, когда база вернётся;
* любое другое исключение - в DLX: неизвестную ошибку безопаснее отложить,
  чем бесконечно перекладывать сообщение по кругу.

Потеря соединения с брокером - тоже временный сбой: consumer переподключается
с нарастающей паузой. Неподтверждённые сообщения брокер при обрыве вернёт в
очередь сам, а повторная доставка уже обработанного отсекается inbox'ом
`consumed_events`. Ошибки канала, закрытого брокером (например, расхождение
топологии), - не временные: consumer падает, чтобы это заметили.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import Callable

import pika
import pika.exceptions
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core import tracing
from app.core.config import get_settings
from app.core.log_setup import configure_logging
from app.db.base import get_session_factory
from app.escalations.consumer import ConsumeResult, PoisonMessageError, handle_escalation_created
from app.messaging.topology import NOTIFY_EXCHANGE, QUEUE, declare_topology
from app.workers.backoff import Backoff

log = logging.getLogger("escalation-consumer")

PREFETCH = 10
TRANSIENT_RETRY_DELAY = 2.0

#: Сбои, после которых имеет смысл переподключиться: соединение потеряно,
#: не установлено или канал оказался в неверном состоянии из-за обрыва.
RECONNECTABLE_ERRORS = (
    pika.exceptions.AMQPConnectionError,
    pika.exceptions.ChannelWrongStateError,
)


def consume(
    session_factory: Callable[[], Session],
    url: str,
    *,
    on_connected: Callable[[], None] = lambda: None,
) -> None:
    """Одна жизнь соединения: подключиться и разбирать очередь до обрыва."""
    connection = pika.BlockingConnection(pika.URLParameters(url))
    try:
        _consume_on(connection, session_factory, on_connected)
    finally:
        # После обрыва соединение уже закрыто: close() бросил бы
        # ConnectionWrongStateError и замаскировал исходную ошибку.
        if connection.is_open:
            with contextlib.suppress(pika.exceptions.AMQPError):
                connection.close()


def run(
    session_factory: Callable[[], Session], url: str, *, backoff: Backoff | None = None
) -> None:
    backoff = backoff or Backoff()

    def on_connected() -> None:
        if backoff.attempt:
            log.info("broker is back after %d failed attempts", backoff.attempt)
        backoff.reset()

    while True:
        try:
            consume(session_factory, url, on_connected=on_connected)
            return
        except RECONNECTABLE_ERRORS as exc:
            delay = backoff.next_delay()
            log.warning(
                "broker connection lost (attempt %d), reconnect in %.1fs: %r",
                backoff.attempt,
                delay,
                exc,
            )
            time.sleep(delay)


def _consume_on(
    connection: pika.BlockingConnection,
    session_factory: Callable[[], Session],
    on_connected: Callable[[], None],
) -> None:
    channel = connection.channel()
    declare_topology(channel)
    # prefetch ограничивает число неподтверждённых сообщений на consumer'а:
    # без него один процесс забрал бы всю очередь, включая приоритетные.
    channel.basic_qos(prefetch_count=PREFETCH)

    def on_message(ch, method, properties, body: bytes) -> None:
        try:
            payload = json.loads(body)
            # trace_id исходного запроса: строки лога consumer'а связываются
            # с HTTP-запросом и записью audit_log (раздел «Наблюдаемость»).
            tracing.set_trace_id(payload.get("trace_id") if isinstance(payload, dict) else None)
            with session_factory() as session:
                result = handle_escalation_created(
                    session, idempotency_key=properties.message_id, payload=payload
                )
        except (PoisonMessageError, ValueError) as exc:
            log.error("poison message → DLX: %s", exc)
            ch.basic_nack(method.delivery_tag, requeue=False)
            return
        except OperationalError as exc:
            log.warning("database unavailable, requeue: %s", exc.orig)
            time.sleep(TRANSIENT_RETRY_DELAY)
            ch.basic_nack(method.delivery_tag, requeue=True)
            return
        except Exception:
            log.exception("unexpected error → DLX")
            ch.basic_nack(method.delivery_tag, requeue=False)
            return

        if result is ConsumeResult.PROCESSED:
            # Уведомление для WS - best effort: оно подсказка консоли, а не
            # источник данных, поэтому без confirm'а и без повторов.
            ch.basic_publish(
                NOTIFY_EXCHANGE,
                routing_key="",
                body=json.dumps(
                    {
                        "type": "escalation.queued",
                        "escalation_id": payload.get("escalation_id"),
                        "ticket_id": payload.get("ticket_id"),
                        "priority": payload.get("priority", 0),
                        "reason": payload.get("reason"),
                    }
                ).encode("utf-8"),
            )
        log.info("%s %s", result.value, properties.message_id)
        # ack строго после commit'а в Postgres (ADR-004).
        ch.basic_ack(method.delivery_tag)

    channel.basic_consume(QUEUE, on_message)
    on_connected()
    log.info("consuming %s, prefetch=%d", QUEUE, PREFETCH)
    channel.start_consuming()


def main() -> None:
    configure_logging(get_settings().log_format)
    with contextlib.suppress(KeyboardInterrupt):
        run(get_session_factory(), get_settings().rabbitmq_url)
    log.info("stopped")


if __name__ == "__main__":
    main()
