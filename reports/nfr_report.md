# Отчёт по нефункциональным требованиям

Сформирован 2026-09-22 скриптом `scripts/nfr_report.py` из артефактов прогонов (`reports/`, `eval/report_aliceai-llm-flash_cross-check.json`) и тестов репозитория. Числа руками не вносятся.

**Итог:** 10 выполнено, 0 частично, 0 не выполнено, 0 не измеримо.

| NFR | Требование | Порог | Замер | Статус |
| --- | --- | --- | --- | --- |
| NFR1 | Задержка | автоответ ≤ 8 с, до оператора ≤ 15 с (p95) | автоответ p95 6.30 с; до очереди оператора p95 6.78 с | **выполнено** |
| NFR2 | Качество классификации | recall high-risk ≥ 0.95, macro-F1 ≥ 0.85 | recall жалоба 0.96, возврат 1.00; macro-F1 0.90 | **выполнено** |
| NFR3 | Проверяемость | 100% решений трассируемы, трейс ≤ 2 с | трейс p95 0.01 с; rule_id, оба confidence, версии документов - в каждом решении | **выполнено** |
| NFR4 | Безопасность и приватность | PII маскируется; хранение ≤ 90 дней; чужой тикет недоступен | маскирование, защита от IDOR и retention проверены; открытые тикеты старше срока не вычищаются, а фиксируются как нарушение | **выполнено** |
| NFR5 | Стоимость на тикет | ≤ $0.01–0.02 переменной стоимости | p95 $0.0013 (0.108 ₽), среднее $0.0009 | **выполнено** |
| NFR6 | Плавная деградация | RAG < порога → эскалация; сбой LLM → retry ×3, затем эскалация | retry ×3 с экспоненциальной паузой в пределах 10 с, без повторов на 4xx; затем эскалация; сбой на черновике не подменяет причину эскалации | **выполнено** |
| NFR7 | Учёт ручных корректировок | 100% правок оператора с diff | draft и final сохраняются в operator_actions, сходство - в audit_log | **выполнено** |
| NFR8 | Масштабируемость | 50 одновременных тикетов без деградации NFR1 | 50: p95 6.30 с, ошибок 0%, соединений с БД ≤ 22; 100: p95 6.41 с, ошибок 0% | **выполнено** |
| NFR9 | Ограничение цикла уточнений | максимум 1 уточнение; таймаут ответа 30 мин | лимит и принудительная эскалация по таймауту проверены | **выполнено** |
| NFR10 | Устойчивость к prompt injection | 0 успешных инъекций | успешных инъекций: 0% из 24 | **выполнено** |

## Как измерено

**NFR1.** нагрузочный тест, 50 одновременных × 3 волны, LLM имитирован по бюджету (2000/4500 мс). До исправлений: p95 автоответа 12.98 с.

**NFR2.** golden set 359 тикетов, классификатор yandex/aliceai-llm-flash (LLM; источник уверенности: cross_check (ADR-009, ADR-012)); отчёт `eval/report_aliceai-llm-flash_cross-check.json`.

**NFR3.** время - нагрузочный тест; полнота трейса - интеграционные тесты.
Тесты: `test_pipeline_integration.py::test_auto_answer_is_persisted_with_full_trace`, `test_pipeline_integration.py::test_retrievals_are_linked_to_document_versions`, `test_audit_api_integration.py::test_audit_shows_both_clarification_iterations`.

**NFR4.** retention вычищает персональные данные закрытых тикетов старше 90 дней, сохраняя audit_log и агрегаты; строки не удаляются, иначе каскад унёс бы аудит. Открытый тикет старше срока - предупреждение шедулера, а не вычистка рабочих данных оператора.
Тесты: `test_retention_integration.py::test_expired_closed_ticket_loses_personal_data`, `test_retention_integration.py::test_audit_and_metrics_survive_scrub`, `test_retention_integration.py::test_open_expired_ticket_is_reported_not_scrubbed`, `test_pipeline_integration.py::test_pii_never_reaches_audit_log`, `test_graph.py::test_pii_is_redacted_before_it_reaches_retriever`, `test_tickets_api_integration.py::test_foreign_ticket_cannot_be_read_with_own_token`, `test_tickets_api_integration.py::test_operator_token_is_not_a_ticket_token`.

**NFR5.** 41 обращений golden set через полный граф агента на yandex/aliceai-llm-flash (k = 1, сверка с базовой линией, BAAI/bge-m3, τ_rag = 0.6); расход - из `usage` ответов провайдера, тем же счётчиком, что пишет SLI в audit_log. По маршрутам: автоответ - 0.097 ₽ (6); без генерации - 0.032 ₽ (16); эскалация с черновиком - 0.098 ₽ (19). Курс 84.0954 ₽/$ (ЦБ РФ) - допущение замера. Доли маршрутов - как в golden set, а не в реальном потоке.

**NFR6.** адаптеры LLM (ADR-009) на имитированном провайдере (httpx.MockTransport, управляемое время); на живом ключе не проверялось.
Тесты: `test_llm_adapters.py::test_retries_are_capped_at_three_then_escalation`, `test_llm_adapters.py::test_retry_respects_overall_deadline`, `test_llm_adapters.py::test_client_errors_are_not_retried`, `test_tickets_api_integration.py::test_llm_outage_escalates_instead_of_failing`, `test_graph.py::test_draft_failure_keeps_escalation_reason`, `test_graph.py::test_low_rag_confidence_escalates_instead_of_answering`.

**NFR7.** интеграционные тесты API консоли.
Тесты: `test_escalations_api_integration.py::test_edit_stores_draft_and_final_for_diff`.

**NFR8.** один процесс API; потолок - пул потоков (API_WORKER_THREADS=100). LLM в нагрузочном тесте имитирован: лимиты RPM/TPM реального провайдера не проверены.

**NFR9.** юнит-тесты Decision Engine и интеграционные тесты шедулера.
Тесты: `test_decision_engine.py::test_invariant_4_clarification_only_on_first_iteration`, `test_scheduler_integration.py::test_silent_client_is_escalated_after_timeout`.

**NFR10.** adversarial-набор; маршрут выбирает код (ADR-001). Проверен на LLM-классификаторе yandex/aliceai-llm-flash.

## Что изменилось по итогам нагрузочного теста

Первый прогон при 50 одновременных тикетах дал p95 автоответа выше порога NFR1. Причины и исправления:

1. `functools.lru_cache` на фабриках не потокобезопасен: первая волна запросов создавала по движку SQLAlchemy на поток, и число соединений с Postgres перестало быть ограниченным. Заменён на `app.core.singleton.once`.
2. Запрос держал соединение с базой всё время ожидания LLM. Теперь соединение возвращается в пул перед каждым долгим шагом (`release_connection`), пул задан явно.
3. Пул потоков anyio по умолчанию - 40, а тикет занимает поток на время ожидания LLM. Размер вынесен в конфиг (`API_WORKER_THREADS`, 100).

## Ограничения замеров

- Задержка LLM в нагрузочном тесте имитирована в пределах бюджета шагов. На реальном провайдере без нагрузки (yandex/aliceai-llm-flash, тикетов подряд: 41) p95 обработки тикета - 1.9 с; под нагрузкой и с лимитами RPM/TPM провайдера не проверялось.
- Качество (NFR2) измерено на yandex/aliceai-llm-flash и синтетическом golden set без доли публичных датасетов; сверка с базовой линией (ADR-012), скорее всего, выглядит на нём лучше, чем будет на реальных обращениях: словарь писался под те же формулировки.
- Стенд однопроцессный и локальный: сеть, отказоустойчивость Postgres и RabbitMQ не нагружались.
