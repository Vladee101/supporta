"""Замер стоимости LLM на тикет (NFR5) на реальном провайдере.

    python -m scripts.measure_cost --rub-per-usd 90 --confirm-cost

Выборка обращений из golden set проходит через полный граф агента - PII,
классификация, поиск, решение, генерация ответа или черновика (ADR-008), - с
провайдером и порогами из конфигурации. Расход берётся из ответов провайдера
(`usage`), тем же счётчиком, что пишет SLI «стоимость на тикет» в audit_log;
ничего не досчитывается по прайсу.

NFR5 задан в долларах, провайдер считает в рублях - курс передаётся явно и
записывается в отчёт: это допущение замера, а не измеренная величина.

Выборка стратифицирована по категориям golden set, поэтому доли маршрутов
(автоответ / эскалация с черновиком / без черновика) - как в golden set, а не
как в реальном потоке. Отчёт даёт стоимость и по маршрутам, чтобы её можно было
пересчитать на реальные доли.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from app.agent.factory import build_classifier_and_generator, get_embedding_provider
from app.agent.graph import TicketGraph
from app.core.config import get_settings, thresholds
from app.db.base import get_session_factory
from app.services.retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"

#: Порог NFR5, $ переменной стоимости на тикет (верхняя граница диапазона).
NFR5_USD = 0.02


def _sample(items: list[dict], size: int, seed: int) -> list[dict]:
    """Пропорционально категориям, минимум одно обращение на категорию."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)
    rng = random.Random(seed)
    sample: list[dict] = []
    for _category, group in sorted(by_category.items()):
        share = max(1, round(size * len(group) / len(items)))
        sample += rng.sample(group, min(share, len(group)))
    return sample


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(_percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Стоимость LLM на тикет (NFR5)")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--rub-per-usd", type=float, required=True, help="курс для сравнения с порогом NFR5 в $"
    )
    parser.add_argument("--confirm-cost", action="store_true", help="подтвердить платный прогон")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if settings.llm_provider == "baseline":
        sys.exit("LLM_PROVIDER=baseline: замерять нечего - базовая линия не тратит")

    items = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()]
    sample = _sample(items, args.sample, args.seed)
    if not args.confirm_cost:
        sys.exit(
            f"{len(sample)} обращений через {settings.llm_provider} "
            f"({settings.llm_classify_model} / {settings.llm_generate_model}): до "
            f"{len(sample) * (settings.llm_k_samples + 2)} платных вызовов. "
            "Запустите с --confirm-cost."
        )

    classifier, generator = build_classifier_and_generator(settings)
    provider = get_embedding_provider()
    graph = TicketGraph(classifier, Retriever(provider), generator, thresholds())

    tickets = []
    with get_session_factory()() as session:
        for item in sample:
            started = time.monotonic()
            outcome = graph.run(session, item["text"])
            usage = outcome.llm_usage or {}
            tickets.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "action": outcome.decision.action.value,
                    "rule_id": outcome.decision.rule_id,
                    "with_draft": outcome.draft is not None,
                    "seconds": round(time.monotonic() - started, 3),
                    **usage,
                }
            )
            print(
                f"  {item['id']:6} {outcome.decision.rule_id:6} вызовов {usage.get('calls')}, "
                f"{usage.get('cost')} ₽"
            )

    unknown = [t for t in tickets if t.get("cost") is None]
    if unknown:
        sys.exit(f"провайдер не сообщил стоимость для {len(unknown)} тикетов - NFR5 не посчитать")

    costs = [t["cost"] for t in tickets]
    per_route: dict[str, list[float]] = defaultdict(list)
    for t in tickets:
        route = (
            "автоответ"
            if t["action"] == "A1"
            else ("эскалация с черновиком" if t["with_draft"] else "без генерации")
        )
        per_route[route].append(t["cost"])

    summary = {
        "cost_rub": _stats(costs),
        "cost_usd": {k: round(v / args.rub_per_usd, 6) for k, v in _stats(costs).items()},
        "calls_per_ticket": _stats([float(t["calls"]) for t in tickets]),
        "tokens_per_ticket": _stats(
            [float(t["prompt_tokens"] + t["completion_tokens"]) for t in tickets]
        ),
        "seconds_per_ticket": _stats([t["seconds"] for t in tickets]),
        "by_route": {
            route: {"tickets": len(values), "mean_cost_rub": round(statistics.fmean(values), 4)}
            for route, values in sorted(per_route.items())
        },
    }
    p95_usd = summary["cost_usd"]["p95"]
    report = {
        "provider": settings.llm_provider,
        "classify_model": settings.llm_classify_model,
        "generate_model": settings.llm_generate_model,
        "k_samples": settings.llm_k_samples,
        "cross_check": settings.llm_cross_check,
        "embedding_provider": provider.model_id,
        "rag_threshold": settings.rag_confidence_threshold,
        "rub_per_usd": args.rub_per_usd,
        "nfr5_usd": NFR5_USD,
        "nfr5_met": p95_usd <= NFR5_USD,
        "sample": len(tickets),
        "seed": args.seed,
        "summary": summary,
        "tickets": tickets,
    }
    model = settings.llm_generate_model.replace("/", "_")
    out = args.out or ROOT / "reports" / f"llm_cost_{model}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rub, usd = summary["cost_rub"], summary["cost_usd"]
    print(
        f"\nстоимость на тикет: среднее {rub['mean']:.3f} ₽ (${usd['mean']:.5f}), "
        f"p95 {rub['p95']:.3f} ₽ (${usd['p95']:.5f}); порог NFR5 ${NFR5_USD}"
    )
    for route, value in summary["by_route"].items():
        print(f"  {route}: {value['tickets']} тикетов, {value['mean_cost_rub']:.3f} ₽")
    print(f"отчёт: {out}")


if __name__ == "__main__":
    main()
