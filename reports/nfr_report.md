# Отчёт по нефункциональным требованиям

Сформирован 2026-09-18 скриптом `scripts/nfr_report.py` из артефактов прогонов (`reports/`, `eval/report.json`) и тестов репозитория. Числа руками не вносятся.

**Итог:** 7 выполнено, 1 частично, 1 не выполнено, 1 не измеримо.

| NFR | Требование | Порог | Замер | Статус |
| --- | --- | --- | --- | --- |
| NFR1 | Задержка | автоответ ≤ 8 с, до оператора ≤ 15 с (p95) | автоответ p95 6.30 с; до очереди оператора p95 6.78 с | **выполнено** |
| NFR2 | Качество классификации | recall high-risk ≥ 0.95, macro-F1 ≥ 0.85 | recall жалоба 0.66, возврат 0.90; macro-F1 0.80 | **не выполнено** |
| NFR3 | Проверяемость | 100% решений трассируемы, трейс ≤ 2 с | трейс p95 0.01 с; rule_id, оба confidence, версии документов - в каждом решении | **выполнено** |
| NFR4 | Безопасность и приватность | PII маскируется; хранение ≤ 90 дней; чужой тикет недоступен | маскирование и защита от IDOR проверены; retention-джоб не реализован | **частично** |
| NFR5 | Стоимость на тикет | ≤ $0.01–0.02 переменной стоимости | — | **не измеримо** |
| NFR6 | Плавная деградация | RAG < порога → эскалация; сбой LLM → retry ×3, затем эскалация | retry ×3 с экспоненциальной паузой в пределах 10 с, без повторов на 4xx; затем эскалация; сбой на черновике не подменяет причину эскалации | **выполнено** |
| NFR7 | Учёт ручных корректировок | 100% правок оператора с diff | draft и final сохраняются в operator_actions, сходство - в audit_log | **выполнено** |
| NFR8 | Масштабируемость | 50 одновременных тикетов без деградации NFR1 | 50: p95 6.30 с, ошибок 0%, соединений с БД ≤ 22; 100: p95 6.41 с, ошибок 0% | **выполнено** |
| NFR9 | Ограничение цикла уточнений | максимум 1 уточнение; таймаут ответа 30 мин | лимит и принудительная эскалация по таймауту проверены | **выполнено** |
| NFR10 | Устойчивость к prompt injection | 0 успешных инъекций | успешных инъекций: 0% из 24 | **выполнено** |

## Как измерено

**NFR1.** нагрузочный тест, 50 одновременных × 3 волны, LLM имитирован по бюджету (2000/4500 мс). До исправлений: p95 автоответа 12.98 с.

**NFR2.** golden set 359 тикетов, классификатор baseline-keywords-v1 (базовая линия: LLM-провайдер не подключён, ADR-009).

**NFR3.** время - нагрузочный тест; полнота трейса - интеграционные тесты.
Тесты: `test_pipeline_integration.py::test_auto_answer_is_persisted_with_full_trace`, `test_pipeline_integration.py::test_retrievals_are_linked_to_document_versions`, `test_audit_api_integration.py::test_audit_shows_both_clarification_iterations`.

**NFR4.** retention сырых тикетов (удаление данных) вынесен из шедулера этапа 4 - требует отдельной проверки.
Тесты: `test_pipeline_integration.py::test_pii_never_reaches_audit_log`, `test_graph.py::test_pii_is_redacted_before_it_reaches_retriever`, `test_tickets_api_integration.py::test_foreign_ticket_cannot_be_read_with_own_token`, `test_tickets_api_integration.py::test_operator_token_is_not_a_ticket_token`.

**NFR5.** переменная стоимость - это LLM-вызовы, а провайдер не выбран (ADR-009). Базовая линия стоит $0; измерение возможно только с реальным провайдером по токенам в трейсе.

**NFR6.** адаптеры LLM (ADR-009) на имитированном провайдере (httpx.MockTransport, управляемое время); на живом ключе не проверялось.
Тесты: `test_llm_adapters.py::test_retries_are_capped_at_three_then_escalation`, `test_llm_adapters.py::test_retry_respects_overall_deadline`, `test_llm_adapters.py::test_client_errors_are_not_retried`, `test_tickets_api_integration.py::test_llm_outage_escalates_instead_of_failing`, `test_graph.py::test_draft_failure_keeps_escalation_reason`, `test_graph.py::test_low_rag_confidence_escalates_instead_of_answering`.

**NFR7.** интеграционные тесты API консоли.
Тесты: `test_escalations_api_integration.py::test_edit_stores_draft_and_final_for_diff`.

**NFR8.** один процесс API; потолок - пул потоков (API_WORKER_THREADS=100). Лимиты RPM/TPM реального LLM-провайдера не проверены - их нет без провайдера.

**NFR9.** юнит-тесты Decision Engine и интеграционные тесты шедулера.
Тесты: `test_decision_engine.py::test_invariant_4_clarification_only_on_first_iteration`, `test_scheduler_integration.py::test_silent_client_is_escalated_after_timeout`.

**NFR10.** adversarial-набор; маршрут выбирает код (ADR-001). Проверен против базовой линии - на LLM-классификаторе прогон нужно повторить.

## Что изменилось по итогам нагрузочного теста

Первый прогон при 50 одновременных тикетах дал p95 автоответа выше порога NFR1. Причины и исправления:

1. `functools.lru_cache` на фабриках не потокобезопасен: первая волна запросов создавала по движку SQLAlchemy на поток, и число соединений с Postgres перестало быть ограниченным. Заменён на `app.core.singleton.once`.
2. Запрос держал соединение с базой всё время ожидания LLM. Теперь соединение возвращается в пул перед каждым долгим шагом (`release_connection`), пул задан явно.
3. Пул потоков anyio по умолчанию - 40, а тикет занимает поток на время ожидания LLM. Размер вынесен в конфиг (`API_WORKER_THREADS`, 100).

## Ограничения замеров

- Задержка LLM имитирована равномерным распределением в пределах бюджета шагов. Хвосты реального провайдера длиннее и зависят от его лимитов - прогон нужно повторить с провайдером.
- Качество (NFR2) измерено на базовой линии и синтетическом golden set без доли публичных датасетов.
- Стенд однопроцессный и локальный: сеть, отказоустойчивость Postgres и RabbitMQ не нагружались.
