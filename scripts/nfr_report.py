"""Отчёт «замер vs порог» по всем NFR из design document.

    python -m scripts.nfr_report

Источники - только артефакты прогонов, никаких чисел руками:

* reports/load_test_fixed.json, load_test_stress100.json, load_test_baseline.json -
  нагрузочный тест (NFR1, NFR3, NFR8);
* eval/report.json - качество на golden set и adversarial-наборе (NFR2, NFR10);
* тесты - для требований, которые проверяются поведением, а не числом. Скрипт
  сверяет, что каждый упомянутый тест существует: отчёт не может сослаться на
  проверку, которой нет.

Статусы: «выполнено», «не выполнено», «частично» (часть требования не
реализована - с указанием какая), «не измеримо» (нужен компонент, которого
нет, - с указанием какой).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TESTS = ROOT / "tests"

PASS, FAIL, PARTIAL, NA = "выполнено", "не выполнено", "частично", "не измеримо"


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"нет артефакта {path.relative_to(ROOT)} - сначала запустите соответствующий прогон"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _tests_exist(refs: list[str]) -> list[str]:
    """Вернуть ссылки на тесты, которых нет в репозитории."""
    missing = []
    for ref in refs:
        file_name, test_name = ref.split("::")
        source = (
            (TESTS / file_name).read_text(encoding="utf-8") if (TESTS / file_name).exists() else ""
        )
        if not re.search(rf"^def {re.escape(test_name)}\(", source, re.MULTILINE):
            missing.append(ref)
    return missing


def _s(value: float | None, unit: str = " с") -> str:
    return "—" if value is None else f"{value:.2f}{unit}"


def build() -> tuple[list[dict], list[str]]:
    fixed = _load(REPORTS / "load_test_fixed.json")
    stress = _load(REPORTS / "load_test_stress100.json")
    baseline = _load(REPORTS / "load_test_baseline.json")
    quality = _load(ROOT / "eval" / "report.json")
    q = quality["metrics"]

    rows: list[dict] = []

    def row(nfr, requirement, threshold, measured, status, method, tests=()):
        rows.append(
            {
                "nfr": nfr,
                "requirement": requirement,
                "threshold": threshold,
                "measured": measured,
                "status": status,
                "method": method,
                "tests": list(tests),
            }
        )

    auto_p95 = fixed["auto_answer"]["p95"]
    queue_p95 = fixed["operator_queue"]["p95"]
    row(
        "NFR1",
        "Задержка",
        "автоответ ≤ 8 с, до оператора ≤ 15 с (p95)",
        f"автоответ p95 {_s(auto_p95)}; до очереди оператора p95 {_s(queue_p95)}",
        PASS if auto_p95 <= 8 and queue_p95 <= 15 else FAIL,
        f"нагрузочный тест, {fixed['config']['concurrency']} одновременных × "
        f"{fixed['config']['waves']} волны, LLM имитирован по бюджету "
        f"({fixed['config']['simulated_llm_classify_ms']}/{fixed['config']['simulated_llm_generate_ms']} мс). "
        f"До исправлений: p95 автоответа {_s(baseline['auto_answer']['p95'])}",
    )

    recall_ok = q["recall_complaint"] >= 0.95 and q["recall_refund"] >= 0.95
    f1_ok = q["macro_f1_regular"] >= 0.85
    row(
        "NFR2",
        "Качество классификации",
        "recall high-risk ≥ 0.95, macro-F1 ≥ 0.85",
        f"recall жалоба {q['recall_complaint']:.2f}, возврат {q['recall_refund']:.2f}; "
        f"macro-F1 {q['macro_f1_regular']:.2f}",
        PASS if recall_ok and f1_ok else FAIL,
        f"golden set {quality['golden_set_size']} тикетов, классификатор {quality['classifier']} "
        "(базовая линия: LLM-провайдер не подключён, ADR-009)",
    )

    audit_p95 = fixed["audit"]["p95"]
    row(
        "NFR3",
        "Проверяемость",
        "100% решений трассируемы, трейс ≤ 2 с",
        f"трейс p95 {_s(audit_p95)}; rule_id, оба confidence, версии документов - в каждом решении",
        PASS if audit_p95 <= 2 else FAIL,
        "время - нагрузочный тест; полнота трейса - интеграционные тесты",
        [
            "test_pipeline_integration.py::test_auto_answer_is_persisted_with_full_trace",
            "test_pipeline_integration.py::test_retrievals_are_linked_to_document_versions",
            "test_audit_api_integration.py::test_audit_shows_both_clarification_iterations",
        ],
    )

    row(
        "NFR4",
        "Безопасность и приватность",
        "PII маскируется; хранение ≤ 90 дней; чужой тикет недоступен",
        "маскирование, защита от IDOR и retention проверены; открытые тикеты старше срока "
        "не вычищаются, а фиксируются как нарушение",
        PASS,
        "retention вычищает персональные данные закрытых тикетов старше 90 дней, сохраняя "
        "audit_log и агрегаты; строки не удаляются, иначе каскад унёс бы аудит. Открытый "
        "тикет старше срока - предупреждение шедулера, а не вычистка рабочих данных оператора",
        [
            "test_retention_integration.py::test_expired_closed_ticket_loses_personal_data",
            "test_retention_integration.py::test_audit_and_metrics_survive_scrub",
            "test_retention_integration.py::test_open_expired_ticket_is_reported_not_scrubbed",
            "test_pipeline_integration.py::test_pii_never_reaches_audit_log",
            "test_graph.py::test_pii_is_redacted_before_it_reaches_retriever",
            "test_tickets_api_integration.py::test_foreign_ticket_cannot_be_read_with_own_token",
            "test_tickets_api_integration.py::test_operator_token_is_not_a_ticket_token",
        ],
    )

    row(
        "NFR5",
        "Стоимость на тикет",
        "≤ $0.01–0.02 переменной стоимости",
        "—",
        NA,
        "переменная стоимость - это LLM-вызовы, а провайдер не выбран (ADR-009). Базовая линия "
        "стоит $0; измерение возможно только с реальным провайдером по токенам в трейсе",
    )

    row(
        "NFR6",
        "Плавная деградация",
        "RAG < порога → эскалация; сбой LLM → retry ×3, затем эскалация",
        "retry ×3 с экспоненциальной паузой в пределах 10 с, без повторов на 4xx; "
        "затем эскалация; сбой на черновике не подменяет причину эскалации",
        PASS,
        "адаптеры LLM (ADR-009) на имитированном провайдере (httpx.MockTransport, управляемое "
        "время); на живом ключе не проверялось",
        [
            "test_llm_adapters.py::test_retries_are_capped_at_three_then_escalation",
            "test_llm_adapters.py::test_retry_respects_overall_deadline",
            "test_llm_adapters.py::test_client_errors_are_not_retried",
            "test_tickets_api_integration.py::test_llm_outage_escalates_instead_of_failing",
            "test_graph.py::test_draft_failure_keeps_escalation_reason",
            "test_graph.py::test_low_rag_confidence_escalates_instead_of_answering",
        ],
    )

    row(
        "NFR7",
        "Учёт ручных корректировок",
        "100% правок оператора с diff",
        "draft и final сохраняются в operator_actions, сходство - в audit_log",
        PASS,
        "интеграционные тесты API консоли",
        ["test_escalations_api_integration.py::test_edit_stores_draft_and_final_for_diff"],
    )

    stress_p95 = stress["auto_answer"]["p95"]
    row(
        "NFR8",
        "Масштабируемость",
        "50 одновременных тикетов без деградации NFR1",
        f"50: p95 {_s(auto_p95)}, ошибок {fixed['error_rate']:.0%}, соединений с БД ≤ "
        f"{fixed['db_connections_max']}; 100: p95 {_s(stress_p95)}, ошибок {stress['error_rate']:.0%}",
        PASS if auto_p95 <= 8 and fixed["error_rate"] == 0 else FAIL,
        "один процесс API; потолок - пул потоков (API_WORKER_THREADS=100). Лимиты RPM/TPM "
        "реального LLM-провайдера не проверены - их нет без провайдера",
    )

    row(
        "NFR9",
        "Ограничение цикла уточнений",
        "максимум 1 уточнение; таймаут ответа 30 мин",
        "лимит и принудительная эскалация по таймауту проверены",
        PASS,
        "юнит-тесты Decision Engine и интеграционные тесты шедулера",
        [
            "test_decision_engine.py::test_invariant_4_clarification_only_on_first_iteration",
            "test_scheduler_integration.py::test_silent_client_is_escalated_after_timeout",
        ],
    )

    row(
        "NFR10",
        "Устойчивость к prompt injection",
        "0 успешных инъекций",
        f"успешных инъекций: {q['injection_success_rate']:.0%} из {quality['adversarial_size']}",
        PASS if q["injection_success_rate"] == 0 else FAIL,
        "adversarial-набор; маршрут выбирает код (ADR-001). Проверен против базовой линии - "
        "на LLM-классификаторе прогон нужно повторить",
    )

    missing = _tests_exist([ref for r in rows for ref in r["tests"]])
    return rows, missing


def render(rows: list[dict]) -> str:
    counts = {
        status: sum(1 for r in rows if r["status"] == status)
        for status in (PASS, PARTIAL, FAIL, NA)
    }
    lines = [
        "# Отчёт по нефункциональным требованиям",
        "",
        f"Сформирован {date.today().isoformat()} скриптом `scripts/nfr_report.py` из артефактов "
        "прогонов (`reports/`, `eval/report.json`) и тестов репозитория. Числа руками не вносятся.",
        "",
        f"**Итог:** {counts[PASS]} выполнено, {counts[PARTIAL]} частично, "
        f"{counts[FAIL]} не выполнено, {counts[NA]} не измеримо.",
        "",
        "| NFR | Требование | Порог | Замер | Статус |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['nfr']} | {r['requirement']} | {r['threshold']} | {r['measured']} | **{r['status']}** |"
        )

    lines += ["", "## Как измерено", ""]
    for r in rows:
        lines.append(f"**{r['nfr']}.** {r['method']}.")
        if r["tests"]:
            lines.append("Тесты: " + ", ".join(f"`{t}`" for t in r["tests"]) + ".")
        lines.append("")

    lines += [
        "## Что изменилось по итогам нагрузочного теста",
        "",
        "Первый прогон при 50 одновременных тикетах дал p95 автоответа выше порога NFR1. "
        "Причины и исправления:",
        "",
        "1. `functools.lru_cache` на фабриках не потокобезопасен: первая волна запросов "
        "создавала по движку SQLAlchemy на поток, и число соединений с Postgres перестало быть "
        "ограниченным. Заменён на `app.core.singleton.once`.",
        "2. Запрос держал соединение с базой всё время ожидания LLM. Теперь соединение "
        "возвращается в пул перед каждым долгим шагом (`release_connection`), пул задан явно.",
        "3. Пул потоков anyio по умолчанию - 40, а тикет занимает поток на время ожидания LLM. "
        "Размер вынесен в конфиг (`API_WORKER_THREADS`, 100).",
        "",
        "## Ограничения замеров",
        "",
        "- Задержка LLM имитирована равномерным распределением в пределах бюджета шагов. "
        "Хвосты реального провайдера длиннее и зависят от его лимитов - прогон нужно повторить "
        "с провайдером.",
        "- Качество (NFR2) измерено на базовой линии и синтетическом golden set без доли "
        "публичных датасетов.",
        "- Стенд однопроцессный и локальный: сеть, отказоустойчивость Postgres и RabbitMQ "
        "не нагружались.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows, missing = build()
    if missing:
        sys.exit("отчёт ссылается на несуществующие тесты: " + ", ".join(missing))

    out = REPORTS / "nfr_report.md"
    out.write_text(render(rows), encoding="utf-8")
    for r in rows:
        print(f"{r['nfr']:6} {r['status']:14} {r['measured']}")
    print(f"\nотчёт: {out}")


if __name__ == "__main__":
    main()
