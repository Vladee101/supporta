"""Процесс outbox-поллера.

    python -m app.workers.outbox_poller

Пока в outbox есть события, пачки идут без паузы; когда очередь пуста -
пауза `--interval`. Запускать можно в нескольких экземплярах: строки
разбираются через FOR UPDATE SKIP LOCKED.
"""

from __future__ import annotations

import argparse
import logging
import time

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.messaging.outbox import DEFAULT_BATCH_SIZE, publish_pending
from app.messaging.publisher import PikaPublisher

log = logging.getLogger("outbox-poller")

#: Порог SLI «возраст неопубликованных outbox_events» из раздела «Наблюдаемость».
STALE_AFTER_SECONDS = 120


def main() -> None:
    parser = argparse.ArgumentParser(description="Outbox poller")
    parser.add_argument("--interval", type=float, default=1.0, help="пауза при пустом outbox, сек")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    publisher = PikaPublisher(get_settings().rabbitmq_url)
    session_factory = get_session_factory()
    log.info("started, batch=%d, interval=%.1fs", args.batch_size, args.interval)

    try:
        while True:
            with session_factory() as session:
                result = publish_pending(session, publisher, batch_size=args.batch_size)

            if result.published or result.failed:
                log.info("published=%d failed=%d", result.published, result.failed)
            if result.stuck:
                log.error("stuck events (attempts exhausted): %d - см. runbook", result.stuck)
            if (result.oldest_pending_age_seconds or 0) > STALE_AFTER_SECONDS:
                log.warning("oldest pending event age: %.0fs", result.oldest_pending_age_seconds)

            if result.failed:
                time.sleep(max(args.interval, 5.0))  # брокер недоступен - не долбим
            elif not result.published:
                # В простое держим соединение живым, иначе первая эскалация после
                # паузы упрётся в закрытое брокером соединение.
                publisher.keepalive()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
