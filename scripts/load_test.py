"""Нагрузочный тест NFR8: 50 одновременных тикетов без деградации NFR1.

    python -m scripts.load_test --env-file loadtest.env --base-url http://127.0.0.1:8100

Что меряется и против какого порога:

* задержка ответа на `POST /tickets` для автоответов - NFR1, ≤ 8 сек (p95);
* время от создания тикета до появления эскалации в очереди оператора
  (по `audit_log.queued_for_operator`) - NFR1 для черновика оператору, ≤ 15 сек;
* доля ошибок - любой не-2xx ответ под нагрузкой считается деградацией;
* время ответа audit trail - NFR3, ≤ 2 сек (p95).

Трафик - смесь категорий в пропорциях, близких к golden set. Каждая волна -
`concurrency` одновременных запросов («50 одновременных тикетов» из NFR8),
между волнами пауза. Результат пишется в reports/load_test.json и читается
генератором отчёта по NFR.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPORTS = Path(__file__).resolve().parent.parent / "reports"

MIX: list[tuple[float, str, list[str]]] = [
    (
        0.45,
        "faq",
        [
            "Подскажите, какие способы оплаты доступны?",
            "Сколько стоит доставка курьером?",
            "Какие сроки доставки по Москве?",
            "Как восстановить пароль от личного кабинета?",
        ],
    ),
    (
        0.25,
        "high_risk",
        [
            "Требую вернуть деньги за бракованный товар",
            "Пишу претензию: курьер нахамил и повредил коробку",
        ],
    ),
    (
        0.15,
        "tech",
        [
            "Приложение вылетает при открытии корзины",
            "Не приходит SMS с кодом подтверждения",
        ],
    ),
    (
        0.15,
        "order",
        [
            "Где мой заказ 1042315? Оформлял пять дней назад",
        ],
    ),
]

THRESHOLDS = {
    "auto_answer_p95_s": 8.0,
    "operator_queue_p95_s": 15.0,
    "error_rate": 0.0,
    "audit_p95_s": 2.0,
}


def percentile(values: list[float], q: float) -> float | None:
    """Ближайший ранг: без интерполяции, чтобы p95 был реальным наблюдением."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(q * len(ordered) + 0.5) - 1))
    return ordered[index]


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def pick(rng: random.Random) -> tuple[str, str]:
    roll, acc = rng.random(), 0.0
    for share, kind, texts in MIX:
        acc += share
        if roll <= acc:
            return kind, rng.choice(texts)
    return MIX[-1][1], MIX[-1][2][0]


async def send(client, secret: str, kind: str, text: str) -> dict:
    from app.core.auth import webhook_signature

    body = json.dumps(
        {"channel": "web", "external_id": f"load-{uuid.uuid4().hex}", "content": text},
        ensure_ascii=False,
    ).encode("utf-8")
    started = time.perf_counter()
    try:
        response = await client.post(
            "/api/v1/tickets",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": webhook_signature(body, secret),
            },
        )
        elapsed = time.perf_counter() - started
        payload = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {
            "kind": kind,
            "http": response.status_code,
            "latency_s": elapsed,
            "status": payload.get("status"),
            "ticket_id": payload.get("ticket_id"),
            "error": None if response.is_success else payload.get("error", {}).get("code"),
        }
    except Exception as exc:  # noqa: BLE001 - любая сетевая ошибка под нагрузкой - это результат
        return {
            "kind": kind,
            "http": None,
            "latency_s": time.perf_counter() - started,
            "status": None,
            "ticket_id": None,
            "error": type(exc).__name__,
        }


async def run_waves(
    base_url: str, concurrency: int, waves: int, pause: float, seed: int
) -> list[dict]:
    import httpx

    from app.core.config import get_settings

    secret = get_settings().channel_secrets["web"]
    rng = random.Random(seed)
    results: list[dict] = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=120, limits=limits) as client:
        for wave in range(1, waves + 1):
            batch = [pick(rng) for _ in range(concurrency)]
            started = time.perf_counter()
            wave_results = await asyncio.gather(*(send(client, secret, k, t) for k, t in batch))
            print(
                f"волна {wave}: {concurrency} запросов за {time.perf_counter() - started:.1f} с, "
                f"ошибок {sum(1 for r in wave_results if r['error'])}"
            )
            results.extend(wave_results)
            if wave < waves:
                await asyncio.sleep(pause)
    return results


class ConnectionSampler:
    """Сколько соединений с базой стенда открыто одновременно (pg_stat_activity).

    Отдельным соединением и отдельным потоком: иначе сэмплер сам занимал бы
    место в пуле, который меряет.
    """

    def __init__(self, database_url: str, interval: float = 0.2) -> None:
        import threading

        from sqlalchemy.engine import make_url

        url = make_url(database_url)
        self._dsn = (
            f"host={url.host} port={url.port} user={url.username} "
            f"password={url.password} dbname=postgres"
        )
        self._database = url.database
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[int] = []

    def _run(self) -> None:
        import psycopg

        with psycopg.connect(self._dsn, autocommit=True) as connection:
            while not self._stop.is_set():
                count = connection.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = %s",
                    (self._database,),
                ).fetchone()[0]
                self.samples.append(count)
                self._stop.wait(self._interval)

    def __enter__(self) -> ConnectionSampler:
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def queue_latencies(ticket_ids: list[str], timeout_s: float) -> tuple[list[float], int]:
    """Время «тикет создан → эскалация в очереди оператора» по audit_log."""
    from sqlalchemy import select

    from app.db.base import get_session_factory
    from app.db.models import AuditLog, Escalation, Ticket

    ids = [uuid.UUID(tid) for tid in ticket_ids]
    deadline = time.monotonic() + timeout_s
    with get_session_factory()() as session:
        while True:
            escalated = set(
                session.scalars(select(Escalation.ticket_id).where(Escalation.ticket_id.in_(ids)))
            )
            queued = {
                row.ticket_id: row.created_at
                for row in session.execute(
                    select(AuditLog.ticket_id, AuditLog.created_at).where(
                        AuditLog.ticket_id.in_(ids), AuditLog.action == "queued_for_operator"
                    )
                )
            }
            if escalated <= set(queued) or time.monotonic() > deadline:
                break
            session.rollback()
            time.sleep(1)

        created = {
            row.id: row.created_at
            for row in session.execute(
                select(Ticket.id, Ticket.created_at).where(Ticket.id.in_(queued))
            )
        }
    latencies = [(queued[tid] - created[tid]).total_seconds() for tid in queued]
    return latencies, len(escalated - set(queued))


async def audit_latencies(base_url: str, ticket_ids: list[str], samples: int) -> list[float]:
    import httpx
    from sqlalchemy import select

    from app.core.auth import issue_token
    from app.db.base import get_session_factory
    from app.db.models import Operator
    from app.domain.enums import OperatorRole

    with get_session_factory()() as session:
        operator = session.scalar(
            select(Operator).where(Operator.email == "loadtest@support.local")
        )
        if operator is None:
            operator = Operator(
                name="Нагрузочный тест", email="loadtest@support.local", role="operator"
            )
            session.add(operator)
            session.commit()
        token = issue_token(operator.id, OperatorRole.OPERATOR)

    latencies: list[float] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        for ticket_id in ticket_ids[:samples]:
            started = time.perf_counter()
            response = await client.get(
                f"/api/v1/tickets/{ticket_id}/audit", headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            latencies.append(time.perf_counter() - started)
    return latencies


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Нагрузочный тест NFR8")
    parser.add_argument("--env-file", required=True, help="конфигурация стенда (база, vhost)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--pause", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    # До импорта приложения: скрипт читает ту же базу, что и нагружаемый стенд.
    load_dotenv(args.env_file, override=True)
    from app.core.config import get_settings

    settings = get_settings()
    started_at = datetime.now().isoformat(timespec="seconds")

    with ConnectionSampler(settings.database_url) as sampler:
        results = asyncio.run(
            run_waves(args.base_url, args.concurrency, args.waves, args.pause, args.seed)
        )
    ok = [r for r in results if not r["error"]]
    auto = [r["latency_s"] for r in ok if r["status"] == "resolved_auto"]
    ticket_ids = [r["ticket_id"] for r in ok if r["ticket_id"]]

    queue, not_delivered = queue_latencies(ticket_ids, timeout_s=90)
    audit = asyncio.run(audit_latencies(args.base_url, ticket_ids, samples=30))

    errors: dict[str, int] = {}
    for r in results:
        if r["error"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1

    report = {
        "label": args.label,
        "started_at": started_at,
        "config": {
            "concurrency": args.concurrency,
            "waves": args.waves,
            "simulated_llm_classify_ms": settings.simulated_llm_classify_ms,
            "simulated_llm_generate_ms": settings.simulated_llm_generate_ms,
            "embedding_provider": settings.embedding_provider,
        },
        "requests": len(results),
        "error_rate": round(sum(1 for r in results if r["error"]) / max(len(results), 1), 4),
        "errors": errors,
        "all_requests": summary([r["latency_s"] for r in ok]),
        "auto_answer": summary(auto),
        "operator_queue": summary(queue),
        "not_delivered_to_queue": not_delivered,
        # Соединения всего стенда: API + поллер + consumer + сам тест.
        "db_connections_max": max(sampler.samples, default=None),
        "audit": summary(audit),
        "by_kind": {
            kind: summary([r["latency_s"] for r in ok if r["kind"] == kind])
            for kind in {r["kind"] for r in results}
        },
        "thresholds": THRESHOLDS,
    }

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"load_test_{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    print()
    print(f"{'метрика':34} {'порог':>8} {'p50':>7} {'p95':>7} {'max':>7}")
    for name, key, threshold in (
        ("автоответ, сек", "auto_answer", THRESHOLDS["auto_answer_p95_s"]),
        ("до очереди оператора, сек", "operator_queue", THRESHOLDS["operator_queue_p95_s"]),
        ("audit trail, сек", "audit", THRESHOLDS["audit_p95_s"]),
    ):
        s = report[key]
        print(
            f"{name:34} {threshold:8.2f} {fmt(s['p50']):>7} {fmt(s['p95']):>7} {fmt(s['max']):>7}"
        )
    print(f"{'доля ошибок':34} {0.0:8.2f} {report['error_rate']:7.2%}  {errors or ''}")
    print(f"не дошли до очереди оператора: {not_delivered}")
    print(f"максимум соединений с базой стенда: {report['db_connections_max']}")
    print(f"\nотчёт: {out}")


if __name__ == "__main__":
    main()
