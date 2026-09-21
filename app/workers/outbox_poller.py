"""Процесс outbox-поллера.

    python -m app.workers.outbox_poller

Пока в outbox есть события, пачки идут без паузы; когда очередь пуста -
пауза `--interval`. Запускать можно в нескольких экземплярах: строки
разбираются через FOR UPDATE SKIP LOCKED.

Недоступность брокера `publish_pending` переживает сам: событие остаётся в
outbox с увеличенным счётчиком попыток. Недоступность Postgres - временный
сбой, который поллер пережидает с нарастающей паузой, а не падает.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.messaging.outbox import DEFAULT_BATCH_SIZE, publish_pending
from app.messaging.publisher import PikaPublisher
from app.workers.backoff import Backoff

log = logging.getLogger("outbox-poller")

#: Порог SLI «возраст неопубликованных outbox_events» из раздела «Наблюдаемость».
STALE_AFTER_SECONDS = 120


def run(
    session_factory: Callable[[], Session],
    publisher: PikaPublisher,
    *,
    interval: float,
    batch_size: int,
    backoff: Backoff | None = None,
) -> None:
    backoff = backoff or Backoff()
    while True:
        try:
            with session_factory() as session:
                result = publish_pending(session, publisher, batch_size=batch_size)
        except OperationalError as exc:
            delay = backoff.next_delay()
            log.warning(
                "database unavailable (attempt %d), retry in %.1fs: %s",
                backoff.attempt,
                delay,
                exc.orig,
            )
            time.sleep(delay)
            continue
        if backoff.attempt:
            log.info("database is back after %d failed attempts", backoff.attempt)
            backoff.reset()

        if result.published or result.failed:
            log.info("published=%d failed=%d", result.published, result.failed)
        if result.stuck:
            log.error("stuck events (attempts exhausted): %d - см. runbook", result.stuck)
        if (result.oldest_pending_age_seconds or 0) > STALE_AFTER_SECONDS:
            log.warning("oldest pending event age: %.0fs", result.oldest_pending_age_seconds)

        if result.failed:
            time.sleep(max(interval, 5.0))  # брокер недоступен - не долбим
        elif not result.published:
            # В простое держим соединение живым, иначе первая эскалация после
            # паузы упрётся в закрытое брокером соединение.
            publisher.keepalive()
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Outbox poller")
    parser.add_argument("--interval", type=float, default=1.0, help="пауза при пустом outbox, сек")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    publisher = PikaPublisher(get_settings().rabbitmq_url)
    log.info("started, batch=%d, interval=%.1fs", args.batch_size, args.interval)
    try:
        run(get_session_factory(), publisher, interval=args.interval, batch_size=args.batch_size)
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
