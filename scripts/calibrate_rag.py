"""Калибровка порога RAG по процедуре из раздела «Confidence и пороги».

    python -m scripts.calibrate_rag --provider bge

Процедура в документе: порог - минимальный, при котором доля автоответов «без
опоры на найденный документ» падает ниже 5% (по ручной разметке 100 ответов).
Ручной разметки нет, поэтому опора размечена автоматически:

* обращение из golden set «опирается» на выдачу, если среди top-k есть один из
  ожидаемых документов (`expected_slugs`);
* обращение из `eval/out_of_kb.jsonl` не опирается ни на что - в базе знаний нет
  ответа на этот вопрос, и автоответ по нему был бы выдумкой.

Второй набор обязателен: в golden set у всех содержательных обращений есть
ожидаемый документ, и без вопросов вне базы знаний любой порог выглядел бы
безопасным. Порог RAG нужен именно для того, чтобы отсекать такие вопросы.

В расчёт идут только категории, где rag_confidence решает маршрут (faq и
tech_issue - R4/R6): high-risk категории и статус заказа эскалируются
независимо от выдачи. Классификатор не участвует - калибруется поиск.

Порог зависит от embedding-провайдера: косинусная шкала не переносится между
моделями, поэтому отчёт пишется на провайдера.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.db.base import get_session_factory
from app.services.embeddings import HashingEmbeddingProvider, get_embedding_provider
from app.services.pii import redact
from app.services.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
GOLDEN_PATH = EVAL_DIR / "golden_set.jsonl"
OUT_OF_KB_PATH = EVAL_DIR / "out_of_kb.jsonl"

#: Категории, где маршрут решает rag_confidence (R4 - автоответ, R6 - эскалация).
RAG_ROUTED = ("faq", "tech_issue")
#: Допустимая доля автоответов без опоры - из раздела «Confidence и пороги».
MAX_UNGROUNDED = 0.05


def _load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"нет файла {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sweep(samples: list[dict], thresholds: list[float]) -> list[dict]:
    grounded_total = sum(1 for s in samples if s["grounded"])
    rows = []
    for threshold in thresholds:
        passing = [
            s
            for s in samples
            if s["rag_confidence"] is not None and s["rag_confidence"] >= threshold
        ]
        ungrounded = sum(1 for s in passing if not s["grounded"])
        rows.append(
            {
                "threshold": threshold,
                "auto_answer_candidates": len(passing),
                "ungrounded": ungrounded,
                "ungrounded_share": round(ungrounded / len(passing), 4) if passing else 0.0,
                "out_of_kb_passed": sum(1 for s in passing if s["out_of_kb"]),
                # Сколько обращений, на которые в базе есть ответ, порог пропускает.
                "grounded_coverage": (
                    round(sum(1 for s in passing if s["grounded"]) / grounded_total, 4)
                    if grounded_total
                    else 0.0
                ),
            }
        )
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Калибровка порога RAG")
    parser.add_argument("--provider", choices=("auto", "bge", "hashing"), default="auto")
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить отчёт")
    args = parser.parse_args()

    settings = get_settings()
    if args.provider == "hashing":
        provider = HashingEmbeddingProvider()
    else:
        provider = get_embedding_provider()
        if args.provider == "bge" and isinstance(provider, HashingEmbeddingProvider):
            sys.exit('bge-m3 недоступен: pip install -e ".[embeddings]"')
    retriever = Retriever(provider, top_k=settings.rag_top_k)

    items = [g for g in _load(GOLDEN_PATH) if g["category"] in RAG_ROUTED]
    items += _load(OUT_OF_KB_PATH)

    samples = []
    with get_session_factory()() as session:
        for item in items:
            result = retriever.retrieve(session, redact(item["text"]).text)
            found = {chunk.slug for chunk in result.chunks}
            samples.append(
                {
                    "id": item["id"],
                    "rag_confidence": result.rag_confidence,
                    "grounded": bool(set(item.get("expected_slugs") or []) & found),
                    "out_of_kb": bool(item.get("out_of_kb")),
                }
            )

    thresholds = [round(0.05 + 0.01 * step, 2) for step in range(91)]
    rows = _sweep(samples, thresholds)
    safe = [
        r for r in rows if r["auto_answer_candidates"] and r["ungrounded_share"] <= MAX_UNGROUNDED
    ]
    chosen = safe[0] if safe else None

    in_kb = [s["rag_confidence"] for s in samples if not s["out_of_kb"] and s["rag_confidence"]]
    out_kb = [s["rag_confidence"] for s in samples if s["out_of_kb"] and s["rag_confidence"]]
    print(
        f"провайдер: {provider.model_id}; обращений: {len(samples)} "
        f"(в базе знаний: {len(in_kb)}, вне базы: {len(out_kb)})"
    )
    median_in, median_out = sorted(in_kb)[len(in_kb) // 2], sorted(out_kb)[len(out_kb) // 2]
    print(
        f"rag_confidence в базе: min {min(in_kb):.3f}, медиана {median_in:.3f}; "
        f"вне базы: max {max(out_kb):.3f}, медиана {median_out:.3f}\n"
    )
    header = ("порог", "кандидатов", "без опоры", "доля", "вне базы", "покрытие")
    print("{:>6} {:>11} {:>10} {:>7} {:>9} {:>9}".format(*header))
    for r in rows:
        if round(r["threshold"] * 100) % 5 == 0 or r is chosen:
            mark = "  ← минимальный безопасный" if r is chosen else ""
            print(
                f"{r['threshold']:6.2f} {r['auto_answer_candidates']:11d} {r['ungrounded']:10d} "
                f"{r['ungrounded_share']:7.1%} {r['out_of_kb_passed']:9d} "
                f"{r['grounded_coverage']:9.1%}{mark}"
            )

    report = {
        "embedding_provider": provider.model_id,
        "retrieval_mode": settings.rag_retrieval_mode,
        "top_k": settings.rag_top_k,
        "max_ungrounded_share": MAX_UNGROUNDED,
        "samples": len(samples),
        "out_of_kb_samples": len(out_kb),
        "chosen": chosen,
        "sweep": rows,
    }
    out = args.out or EVAL_DIR / f"rag_calibration_{provider.model_id.replace('/', '_')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nотчёт: {out}")
    if chosen is None:
        sys.exit("ни один порог не удерживает долю автоответов без опоры в пределах 5%")


if __name__ == "__main__":
    main()
