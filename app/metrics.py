"""Метрики SLI из раздела «Наблюдаемость» (Prometheus, эндпоинт /metrics).

Два вида метрик, и это разделение намеренное:

* **события** - счётчики и гистограммы в коде: решения агента по правилам,
  задержка обработки, причины эскалаций, расход LLM, действия операторов.
  Доли и скорости считает PromQL (`rate`, отношения счётчиков), а не приложение;
* **состояние** - снимается в момент опроса из первоисточника: возраст
  неопубликованных событий outbox (Postgres - source of truth, ADR-004) и
  глубина очереди и DLX (RabbitMQ). Счётчик в воркере здесь соврал бы: поллер,
  который «встал», перестаёт и обновлять свои метрики, а алерт должен сработать
  именно тогда.

Процесс API с несколькими воркерами uvicorn держит счётчики в каждом процессе
отдельно; для такого запуска нужен multiprocess-режим prometheus_client
(PROMETHEUS_MULTIPROC_DIR). Стенд MVP - один процесс.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pika
import pika.exceptions
from prometheus_client import Counter, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from sqlalchemy import func, select

from app.domain.enums import Action

log = logging.getLogger(__name__)

TICKETS = Counter(
    "support_tickets_processed_total",
    "Обработанные агентом проходы тикета по действию и правилу decision table",
    ["action", "rule_id"],
)
TICKET_SECONDS = Histogram(
    "support_ticket_processing_seconds",
    "Время обработки прохода тикета агентом (NFR1: p95 автоответа ≤ 8 с)",
    ["action"],
    buckets=(0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 30),
)
ESCALATIONS = Counter(
    "support_escalations_total",
    "Эскалации по причине (llm_unavailable - SLI деградации провайдера)",
    ["reason"],
)
LLM_CALLS = Counter("support_llm_calls_total", "Вызовы LLM-провайдера", ["model"])
LLM_TOKENS = Counter("support_llm_tokens_total", "Токены LLM", ["kind"])
LLM_COST = Counter(
    "support_llm_cost_rub_total", "Стоимость вызовов LLM по ответам провайдера, ₽ (NFR5)"
)
LLM_CALLS_WITHOUT_COST = Counter(
    "support_llm_calls_without_cost_total",
    "Вызовы, по которым провайдер не сообщил стоимость: NFR5 по ним не посчитать",
)
OPERATOR_ACTIONS = Counter(
    "support_operator_actions_total",
    "Ответы операторов по типу (edit / всё - operator override rate)",
    ["action_type"],
)


def observe_ticket(outcome, seconds: float) -> None:
    """Записать проход тикета. Вызывается после commit'а - откат не должен попасть в метрики."""
    decision = outcome.decision
    TICKETS.labels(decision.action.value, decision.rule_id).inc()
    TICKET_SECONDS.labels(decision.action.value).observe(seconds)
    if decision.action in (Action.ESCALATE, Action.PRIORITY_ESCALATE) and decision.reason:
        ESCALATIONS.labels(decision.reason.value).inc()

    usage = outcome.llm_usage or {}
    for model, calls in (usage.get("by_model") or {}).items():
        LLM_CALLS.labels(model).inc(calls)
    LLM_TOKENS.labels("prompt").inc(usage.get("prompt_tokens") or 0)
    LLM_TOKENS.labels("completion").inc(usage.get("completion_tokens") or 0)
    if usage.get("cost") is not None:
        LLM_COST.inc(usage["cost"])
    LLM_CALLS_WITHOUT_COST.inc(usage.get("calls_without_cost") or 0)


def observe_llm_unavailable() -> None:
    """Провайдер недоступен после повторов (NFR6): тикет эскалирован в обход графа."""
    TICKETS.labels(Action.ESCALATE.value, "NFR6").inc()
    ESCALATIONS.labels("llm_unavailable").inc()


def observe_operator_action(action_type: str) -> None:
    OPERATOR_ACTIONS.labels(action_type).inc()


class StateCollector(Collector):
    """Состояние outbox и очередей - на момент опроса, из первоисточника."""

    def __init__(self, session_factory, rabbitmq_url: str) -> None:
        self._session_factory = session_factory
        self._rabbitmq_url = rabbitmq_url

    def collect(self):
        yield from self._outbox()
        yield from self._queues()

    def _outbox(self):
        from app.db.models import OutboxEvent
        from app.messaging.outbox import MAX_ATTEMPTS

        pending = GaugeMetricFamily(
            "support_outbox_pending", "Неопубликованные события outbox в работе поллера"
        )
        stuck = GaugeMetricFamily(
            "support_outbox_stuck", "События outbox с исчерпанными попытками - разбор по runbook"
        )
        oldest = GaugeMetricFamily(
            "support_outbox_oldest_pending_seconds",
            "Возраст самого старого неопубликованного события (> 120 с - поллер встал)",
        )
        up = GaugeMetricFamily("support_database_up", "Postgres доступен для снятия метрик")
        try:
            with self._session_factory() as session:
                count, first = session.execute(
                    select(func.count(), func.min(OutboxEvent.created_at)).where(
                        OutboxEvent.published.is_(False), OutboxEvent.attempts < MAX_ATTEMPTS
                    )
                ).one()
                stuck_count = session.scalar(
                    select(func.count()).where(
                        OutboxEvent.published.is_(False), OutboxEvent.attempts >= MAX_ATTEMPTS
                    )
                )
        except Exception as exc:  # noqa: BLE001 - недоступность базы - тоже значение метрики
            log.warning("метрики outbox не сняты: %s", exc)
            up.add_metric([], 0)
            yield up
            return
        up.add_metric([], 1)
        pending.add_metric([], count)
        stuck.add_metric([], stuck_count or 0)
        age = (datetime.now(UTC) - first).total_seconds() if first is not None else 0.0
        oldest.add_metric([], age)
        yield from (up, pending, stuck, oldest)

    def _queues(self):
        from app.messaging.topology import DEAD_LETTER_QUEUE, QUEUE

        depth = GaugeMetricFamily(
            "support_rabbitmq_queue_messages", "Сообщений в очереди", labels=["queue"]
        )
        up = GaugeMetricFamily("support_rabbitmq_up", "RabbitMQ доступен для снятия метрик")
        parameters = pika.URLParameters(self._rabbitmq_url)
        parameters.socket_timeout = 2
        parameters.blocked_connection_timeout = 2
        try:
            connection = pika.BlockingConnection(parameters)
            try:
                channel = connection.channel()
                for queue in (QUEUE, DEAD_LETTER_QUEUE):
                    # passive: только прочитать, не создавать - иначе опрос метрик
                    # мог бы объявить очередь с другими аргументами, чем у воркеров.
                    try:
                        result = channel.queue_declare(queue, passive=True)
                    except pika.exceptions.ChannelClosedByBroker as exc:
                        if exc.reply_code != 404:
                            raise
                        # Очереди ещё нет (consumer не запускался) - это не
                        # недоступность брокера; брокер закрыл канал, нужен новый.
                        channel = connection.channel()
                        continue
                    depth.add_metric([queue], result.method.message_count)
            finally:
                if connection.is_open:
                    connection.close()
        except pika.exceptions.AMQPError as exc:
            log.warning("метрики RabbitMQ не сняты: %r", exc)
            up.add_metric([], 0)
            yield up
            return
        up.add_metric([], 1)
        yield from (up, depth)
