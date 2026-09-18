"""Шедулер: таймауты жизненного цикла эскалаций.

    python -m app.workers.scheduler

Две задачи, обе идемпотентны и безопасны при нескольких экземплярах
(строки берутся через FOR UPDATE SKIP LOCKED):

* эскалация тикетов, по которым клиент не ответил на уточнение (NFR9);
* возврат в очередь эскалаций с истёкшим claim'ом (FR10).

Retention сырых тикетов (NFR4) сюда пока не входит - это отдельная задача
с удалением данных, и она требует своей проверки.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.escalations.operations import release_expired
from app.escalations.timeouts import escalate_clarification_timeouts

log = logging.getLogger("scheduler")


def run_once() -> tuple[int, int]:
    settings = get_settings()
    now = datetime.now(UTC)
    session_factory = get_session_factory()

    with session_factory() as session:
        timed_out = escalate_clarification_timeouts(
            session, now=now, timeout=timedelta(minutes=settings.clarification_timeout_minutes)
        )
    with session_factory() as session:
        released = release_expired(session, now=now)
    return timed_out, released


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="один проход и выход")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        while True:
            timed_out, released = run_once()
            if timed_out or released:
                log.info("clarification timeouts=%d, expired claims=%d", timed_out, released)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
