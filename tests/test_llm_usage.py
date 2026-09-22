"""Учёт расхода LLM на тикет (NFR5): счётчик, адаптеры, граф."""

from __future__ import annotations

from app.agent.graph import TicketGraph
from app.domain.decision import Thresholds
from app.services import llm_usage
from app.services.classifier import BaselineClassifier
from app.services.generation import TemplateResponseGenerator
from tests.fakes import StubRetriever, make_chunk
from tests.test_llm_adapters import LABELS, Recorder, oa_label, openai_client

USAGE = {"prompt_tokens": 200, "completion_tokens": 3, "cost": 0.03}


def test_record_outside_meter_is_a_noop():
    llm_usage.record("m", USAGE)  # не должно падать и никуда не писать


def test_meter_sums_calls_tokens_and_cost():
    with llm_usage.metered() as meter:
        llm_usage.record("classify", USAGE)
        llm_usage.record("generate", {"prompt_tokens": 900, "completion_tokens": 150, "cost": 0.16})

    snapshot = meter.snapshot()
    assert snapshot["calls"] == 2
    assert snapshot["prompt_tokens"] == 1100
    assert snapshot["completion_tokens"] == 153
    assert snapshot["cost"] == 0.19
    assert snapshot["by_model"] == {"classify": 1, "generate": 1}


def test_unknown_cost_is_not_reported_as_free():
    """Провайдер без usage.cost: стоимость неизвестна, а не ноль."""
    with llm_usage.metered() as meter:
        llm_usage.record("m", USAGE)
        llm_usage.record("m", {"prompt_tokens": 10, "completion_tokens": 1})

    assert meter.snapshot()["cost"] is None
    assert meter.snapshot()["calls_without_cost"] == 1


def test_meters_do_not_leak_between_tickets():
    with llm_usage.metered() as first:
        llm_usage.record("m", USAGE)
    with llm_usage.metered() as second:
        pass
    assert first.snapshot()["calls"] == 1
    assert second.snapshot()["calls"] == 0


def test_adapter_records_usage_from_provider_response():
    recorder = Recorder({**oa_label("2"), "usage": USAGE})
    client = openai_client(recorder, confidence_mode="k_sampling", k_samples=1)

    with llm_usage.metered() as meter:
        client.classify("s", "t", LABELS)

    assert meter.snapshot()["calls"] == 1
    assert meter.snapshot()["cost"] == 0.03


def test_k_samples_in_thread_pool_are_all_counted():
    """Выборки k-sampling идут в пуле потоков: без копии контекста они выпали бы из учёта."""
    recorder = Recorder(*[{**oa_label("2"), "usage": USAGE} for _ in range(3)])
    client = openai_client(recorder, confidence_mode="k_sampling", k_samples=3)

    with llm_usage.metered() as meter:
        client.classify("s", "t", LABELS)

    assert meter.snapshot()["calls"] == 3
    assert meter.snapshot()["cost"] == 0.09


class MeteredClassifier(BaselineClassifier):
    def classify(self, text):
        llm_usage.record("classify-model", USAGE)
        return super().classify(text)


class MeteredGenerator(TemplateResponseGenerator):
    def generate(self, ticket_text, chunks):
        llm_usage.record("generate-model", {**USAGE, "cost": 0.2})
        return super().generate(ticket_text, chunks)


def test_graph_collects_usage_of_all_nodes_into_outcome():
    """LangGraph может выполнять узлы в своих потоках - расход всё равно попадает в тикет."""
    graph = TicketGraph(
        MeteredClassifier(),
        StubRetriever((make_chunk(score=0.9),)),
        MeteredGenerator(),
        Thresholds(),
    )
    outcome = graph.run(None, "Подскажите, какие способы оплаты доступны?")

    assert outcome.llm_usage["calls"] == 2
    assert outcome.llm_usage["by_model"] == {"classify-model": 1, "generate-model": 1}
    assert outcome.llm_usage["cost"] == 0.23
