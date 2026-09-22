"""Шедулер: таймауты жизненного цикла эскалаций и retention.

    python -m app.workers.scheduler

Три задачи, все идемпотентны и безопасны при нескольких экземплярах
(строки берутся через FOR UPDATE SKIP LOCKED):

* эскалация тикетов, по которым клиент не ответил на уточнение (NFR9);
* возврат в очередь эскалаций с истёкшим claim'ом (FR10);
* вычистка персональных данных закрытых тикетов старше срока хранения (NFR4).

Недоступность Postgres шедулер пережидает с нарастающей паузой: пропущенный
тик безвреден, следующий подберёт всё, что накопилось.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.core.log_setup import configure_logging
from app.db.base import get_session_factory
from app.escalations.operations import release_expired
from app.escalations.timeouts import escalate_clarification_timeouts
from app.retention import scrub_expired
from app.workers.backoff import Backoff

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

    configure_logging(get_settings().log_format)
    backoff = Backoff()
    try:
        while True:
            try:
                tick = run_once()
            except OperationalError as exc:
                if args.once:
                    raise
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
