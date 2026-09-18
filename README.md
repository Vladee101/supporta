# Support-агент

AI-агент службы поддержки: классификация тикета, RAG-поиск по базе знаний,
**детерминированная** маршрутизация (автоответ / уточнение / эскалация),
human-in-the-loop подтверждение оператором.

Проектная документация - [Support-агент — Design Document.md](Support-агент%20—%20Design%20Document.md).
Код следует документу, а не наоборот: правила маршрутизации, пороги и схема данных
имеют прямые ссылки на разделы документа.

## Быстрый старт

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
cp .env.example .env
docker compose up -d
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m scripts.seed_kb
.venv/Scripts/python.exe -m pytest
```

Расчёт embedding'ов - отдельным шагом, он тянет ~2 ГБ весов bge-m3:

```bash
.venv/Scripts/python.exe -m pip install -e ".[embeddings]"
.venv/Scripts/python.exe -m scripts.index_kb
```

## Структура

```
app/
  domain/decision.py   Decision Engine: R1-R9, R7c, R-default - чистая функция без I/O
  domain/enums.py      категории, действия A1-A4, статусы, причины эскалации
  agent/graph.py       LangGraph: redact → classify → retrieve → decide → act
  agent/service.py     применение решения к БД одной транзакцией (тикет+эскалация+outbox)
  services/pii.py      маскирование PII до первого обращения к LLM (NFR4)
  services/classifier.py   базовая линия + обёртка над LLM-провайдером
  services/embeddings.py   bge-m3 и офлайновый хеширующий провайдер
  services/retrieval.py    pgvector-поиск, rag_confidence = max(relevance_score)
  services/generation.py   автоответ и черновик оператору (ADR-008)
  db/models.py         схема из ERD: история классификаций, версии KB, outbox, claim
  core/config.py       пороги и таймауты (в конфиге, а не в коде правил)
  main.py              FastAPI: /health и просмотр действующей decision table
migrations/            alembic: 0001 - схема + pgvector + HNSW, 0002 - источник confidence
scripts/               seed_kb, index_kb, gen_golden_set, eval
data/kb_seed.json      32 документа базы знаний
eval/                  golden set, adversarial-набор, отчёт метрик
tests/                 юнит-тесты и интеграционные (маркер `integration`)
```

## Decision Engine

Ядро проекта. Маршрут выбирает код, а не LLM (ADR-001), поэтому решение
воспроизводимо, тестируемо и не подвержено prompt injection: текст тикета
физически не может изменить маршрут.

```python
from app.domain.decision import DecisionInput, decide
from app.domain.enums import Category

decide(DecisionInput(Category.REFUND, rag_confidence=0.99, class_confidence=0.99))
# Decision(action=A2, rule_id='R2/R3', reason='high_risk_category', priority=0)
```

Три свойства, ради которых он написан именно так:

- **порядок правил - часть спецификации**: срабатывает первое совпавшее, иначе
  R7b и R8 перекрываются;
- **нет поведения по умолчанию**: непокрытая комбинация даёт `R-default` -
  эскалацию и алерт `decision_table_gap`, а не случайный автоответ клиенту;
- **пороги снаружи**: 0.85 и 0.7 калибруются на golden set и живут в конфиге.

Тесты (`tests/test_decision_engine.py`): по одному каноническому случаю на каждую
строку таблицы, 5 инвариантов из документа, полный перебор 144 комбинаций входа
с проверкой, что ни одна не попадает в `R-default`, границы порогов и проверка,
что недостижимых правил нет.

## Оценка качества

```bash
.venv/Scripts/python.exe -m scripts.gen_golden_set          # golden set: 359 тикетов
.venv/Scripts/python.exe -m scripts.index_kb --provider hashing --reindex
.venv/Scripts/python.exe -m scripts.eval --rag-threshold 0.15
```

`scripts/eval.py` печатает таблицу «метрика / порог / замер / дельта» и сохраняет
`eval/report.json`; с флагом `--gate` возвращает ненулевой код при провале порога -
это и есть гейт для CI из раздела «Методика оценки».

**Как читать цифры.** Сейчас пайплайн работает на базовой линии: словарный
классификатор и хеширующие векторы вместо LLM и bge-m3. Это нижняя граница,
а не результат системы - провайдер LLM не выбран (ADR-009 задаёт критерии
выбора: structured output и logprobs), а веса bge-m3 весят ~2 ГБ и не скачаны.
Порог RAG передаётся флагом, потому что косинусная шкала не переносится между
моделями: 0.7 откалиброван под bge-m3, у хеширующего провайдера верх выдачи
лежит в районе 0.2-0.5.

Известные ограничения golden set: он полностью синтетический (ручное ядро плюс
шаблоны), доли из публичных датасетов в нём нет, поэтому он проверяет поведение
на ожидаемых формулировках, но не устойчивость к реальному языку клиентов.
Groundedness по документу размечается вручную и в автоматический прогон не входит.

## Эскалации (этап 4)

Цепочка процессов:

```
агент ─(одна транзакция)─► escalations + outbox_events
outbox poller ─(confirm)─► RabbitMQ: escalations.created (priority, DLX)
escalation consumer ─► Postgres (inbox + статус) ─► escalations.notify (fanout)
API: мост ─► WebSocket консоли        REST: очередь / контекст / claim / resolve
scheduler ─► таймауты уточнения (NFR9), истёкшие claim'ы (FR10)
```

Запуск (каждый - отдельным процессом):

```bash
.venv/Scripts/python.exe -m app.workers.outbox_poller
.venv/Scripts/python.exe -m app.workers.escalation_consumer
.venv/Scripts/python.exe -m app.workers.scheduler
.venv/Scripts/python.exe -m scripts.issue_token --email anna@example.com --name "Анна"
```

API с мостом уведомлений - `WS_BRIDGE_ENABLED=true` в `.env`.

Гарантии доставки: поллер - at-least-once (публикация подтверждается
publisher confirm'ом до отметки `published`), consumer - дедупликация через
inbox `consumed_events` в той же транзакции, что и смена статуса тикета, ack -
только после commit'а. Сообщение, которое нельзя обработать, уходит в
`escalations.dead`, а не крутится в очереди.

## Консоль оператора (этап 5)

React + TypeScript (Vite), каталог `console/`. API отдаёт сборку по адресу
`/console`:

```bash
cd console && npm install && npm run build
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000   # http://localhost:8000/console/
```

Для разработки - `npm run dev` в `console/` (порт 5173, REST и WebSocket
проксируются на :8000). Вход - по токену оператора из `scripts.issue_token`;
раздел «База знаний» видит только роль `admin`.

Что есть в консоли: очередь с приоритетами и live-обновлением по WebSocket
(при обрыве - опрос по таймеру), карточка эскалации с перепиской, классификацией
и найденными документами (снапшоты - то, что видел агент), claim / ответ /
возврат в очередь, audit trail тикета, CRUD базы знаний с историей версий и
индикатором фоновой индексации.

Попробовать на живом стеке (нужны работающие API, outbox poller и consumer):

```bash
.venv/Scripts/python.exe -m scripts.demo_tickets
```

## Нагрузка и отчёт по NFR (этап 6)

Нагрузочный стенд изолирован: отдельная база `support_load` и vhost `load` в
RabbitMQ (`loadtest.env`), чтобы тестовые тикеты не смешивались с рабочими, а
рабочий consumer не получал чужие события. Задержка LLM имитируется в пределах
бюджета шагов из design document (`SIMULATED_LLM_*_MS`) - без этого базовая
линия отвечает за миллисекунды и тест мерил бы пустую оболочку.

```bash
.venv/Scripts/python.exe -m scripts.serve --env-file loadtest.env --port 8100
.venv/Scripts/python.exe -m scripts.load_test --env-file loadtest.env --label fixed
.venv/Scripts/python.exe -m scripts.nfr_report
```

(поллер и consumer стенда запускаются с `DATABASE_URL` и `RABBITMQ_URL` из `loadtest.env`)

Первый прогон нашёл три дефекта ёмкости: гонку в `lru_cache`-синглтонах (движок
SQLAlchemy на каждый поток первой волны), удержание соединения с базой на время
ожидания LLM и пул потоков anyio в 40 при 50 одновременных тикетах. Результаты до
и после - в [reports/nfr_report.md](reports/nfr_report.md).

## LLM-провайдер (ADR-009)

По умолчанию `LLM_PROVIDER=baseline`: словарный классификатор и шаблонные ответы,
без сети и ключей. Адаптеры в `app/services/llm_adapters`:

- `openai_compatible` - YandexGPT, vLLM и любой API в формате OpenAI Chat Completions;
- `gigachat` - собственный контракт Сбера (OAuth, chat completions v2).

Общее ядро: retry по NFR6 (до трёх повторов в пределах 10 с, 4xx не повторяются),
категории кодируются цифрами, чтобы распределение читалось из `top_logprobs`,
и режим `auto` - logprobs, если провайдер их отдаёт, иначе k-sampling.
Настройки - в `.env.example`, раздел «LLM-провайдер».

Сравнение провайдеров на golden set (платный прогон, поэтому нужен явный флаг):

```bash
.venv/Scripts/python.exe -m scripts.eval --classifier configured --confirm-cost --rag-threshold 0.15
```

Адаптеры проверены на имитированном провайдере; на живом ключе ещё не вызывались.

## Статус по этапам

| Этап | Содержание | Статус |
| --- | --- | --- |
| 0 | Доработка design document (дыра в decision table, ERD, метрики, риски) | готово |
| 1 | Каркас, схема БД, миграции, seed базы знаний | готово |
| 2 | Decision Engine + тесты | готово |
| 3 | Классификация, RAG, генерация, PII-редакция, eval на golden set | готово на базовой линии, LLM-провайдер не подключён |
| 4 | Эскалации: транзакция + outbox + RabbitMQ + consumer + WS | готово |
| 5 | Ingestion API, Operator Console, KB admin | готово |
| 6 | Нагрузочный тест, отчёт «замер vs порог» по NFR | готово - [reports/nfr_report.md](reports/nfr_report.md) |
