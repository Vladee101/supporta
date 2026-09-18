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
"""

from __future__ import annotations

import json
import logging
import time

import pika
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.escalations.consumer import ConsumeResult, PoisonMessageError, handle_escalation_created
from app.messaging.topology import NOTIFY_EXCHANGE, QUEUE, declare_topology

log = logging.getLogger("escalation-consumer")

PREFETCH = 10
TRANSIENT_RETRY_DELAY = 2.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    session_factory = get_session_factory()

    connection = pika.BlockingConnection(pika.URLParameters(get_settings().rabbitmq_url))
    channel = connection.channel()
    declare_topology(channel)
    # prefetch ограничивает число неподтверждённых сообщений на consumer'а:
    # без него один процесс забрал бы всю очередь, включая приоритетные.
    channel.basic_qos(prefetch_count=PREFETCH)

    def on_message(ch, method, properties, body: bytes) -> None:
        try:
            payload = json.loads(body)
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
    log.info("consuming %s, prefetch=%d", QUEUE, PREFETCH)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()
        log.info("stopped")


if __name__ == "__main__":
    main()
