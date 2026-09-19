"""Шедулер: таймауты жизненного цикла эскалаций и retention.

    python -m app.workers.scheduler

Три задачи, все идемпотентны и безопасны при нескольких экземплярах
(строки берутся через FOR UPDATE SKIP LOCKED):

* эскалация тикетов, по которым клиент не ответил на уточнение (NFR9);
* возврат в очередь эскалаций с истёкшим claim'ом (FR10);
* вычистка персональных данных закрытых тикетов старше срока хранения (NFR4).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.escalations.operations import release_expired
from app.escalations.timeouts import escalate_clarification_timeouts
from app.retention import scrub_expired

log = logging.getLogger("scheduler")


@dataclass(frozen=True, slots=True)
class TickResult:
    clarification_timeouts: int
    expired_claims: int
    scrubbed: int
    overdue_open: int


def run_once() -> TickResult:
    settings = get_settings()
    now = datetime.now(UTC)
    session_factory = get_session_factory()

    with session_factory() as session:
        timed_out = escalate_clarification_timeouts(
            session, now=now, timeout=timedelta(minutes=settings.clarification_timeout_minutes)
        )
    with session_factory() as session:
        released = release_expired(session, now=now)
    with session_factory() as session:
        retention = scrub_expired(
            session, now=now, retention=timedelta(days=settings.raw_ticket_retention_days)
        )
    return TickResult(timed_out, released, retention.scrubbed, retention.overdue_open)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="один проход и выход")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        while True:
            tick = run_once()
            if tick.clarification_timeouts or tick.expired_claims or tick.scrubbed:
                log.info(
                    "clarification timeouts=%d, expired claims=%d, retention scrubbed=%d",
                    tick.clarification_timeouts,
                    tick.expired_claims,
                    tick.scrubbed,
                )
            if tick.overdue_open:
                # Открытый тикет старше срока хранения: персональные данные держатся
                # дольше NFR4, потому что тикет ещё в работе. Нужен разбор, не вычистка.
                log.warning("NFR4: открытых тикетов старше срока хранения: %d", tick.overdue_open)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
