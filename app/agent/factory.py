"""Сборка агента из конфигурации.

Провайдер выбирается настройкой `LLM_PROVIDER` (ADR-009): `baseline` -
словарный классификатор и шаблонный генератор без сети, `openai_compatible`
(YandexGPT, vLLM) и `gigachat` - адаптеры из `app.services.llm_adapters`.
Граф и Decision Engine от выбора не зависят.
"""

from __future__ import annotations

import logging

from app.agent.graph import TicketGraph
from app.agent.service import AgentService
from app.core.config import Settings, get_settings, thresholds
from app.core.singleton import once
from app.services.classifier import BaselineClassifier, Classifier, LlmClassifier
from app.services.embeddings import (
    BgeM3EmbeddingProvider,
    EmbeddingProvider,
    HashingEmbeddingProvider,
)
from app.services.generation import (
    LlmResponseGenerator,
    ResponseGenerator,
    TemplateResponseGenerator,
)
from app.services.latency import SlowClassifier, SlowGenerator
from app.services.llm import LLMClient
from app.services.llm_adapters import (
    GigaChatClient,
    GigaChatTokenProvider,
    OpenAICompatibleClient,
    RetryPolicy,
)
from app.services.llm_adapters.openai_compatible import build_http_client
from app.services.retrieval import Retriever

log = logging.getLogger(__name__)


class LLMConfigError(ValueError):
    """Провайдер выбран, но настроен не полностью - падаем при старте, а не на тикете."""


@once
def get_embedding_provider() -> EmbeddingProvider:
    """Один экземпляр на процесс: загрузка bge-m3 стоит секунды и гигабайты."""
    choice = get_settings().embedding_provider
    if choice == "hashing":
        return HashingEmbeddingProvider()
    if choice == "bge":
        return BgeM3EmbeddingProvider()
    try:
        return BgeM3EmbeddingProvider()
    except RuntimeError:
        log.warning("bge-m3 недоступен, используется хеширующий провайдер: качество RAG ниже")
        return HashingEmbeddingProvider()


def build_llm_clients(settings: Settings) -> tuple[LLMClient, LLMClient]:
    """Два клиента - классификация и генерация - на общем HTTP-пуле.

    Модели разные намеренно (ADR-009): дешёвая на 100% тикетов, сильная - только
    там, где нужен текст для клиента или оператора.
    """
    if not settings.llm_classify_model or not settings.llm_generate_model:
        raise LLMConfigError("нужны LLM_CLASSIFY_MODEL и LLM_GENERATE_MODEL")

    http = build_http_client(verify=settings.llm_ca_bundle or True)
    common = {
        "http": http,
        "retry": RetryPolicy(
            max_retries=settings.llm_max_retries,
            deadline_seconds=settings.llm_retry_deadline_seconds,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
        ),
        "confidence_mode": settings.llm_confidence_mode,
        "k_samples": settings.llm_k_samples,
        "sampling_temperature": settings.llm_sampling_temperature,
    }

    if settings.llm_provider == "openai_compatible":
        if not settings.llm_base_url:
            raise LLMConfigError("для openai_compatible нужен LLM_BASE_URL")
        dialect = {
            "base_url": settings.llm_base_url,
            "api_key": settings.llm_api_key,
            "auth_scheme": settings.llm_auth_scheme,
            "choice_constraint": settings.llm_choice_constraint,
        }
        return (
            OpenAICompatibleClient(model=settings.llm_classify_model, **dialect, **common),
            OpenAICompatibleClient(model=settings.llm_generate_model, **dialect, **common),
        )

    if settings.llm_provider == "gigachat":
        if not settings.gigachat_credentials:
            raise LLMConfigError("для gigachat нужен GIGACHAT_CREDENTIALS (ключ авторизации)")
        tokens = GigaChatTokenProvider(
            credentials=settings.gigachat_credentials,
            http=http,
            scope=settings.gigachat_scope,
            auth_url=settings.gigachat_auth_url,
        )
        dialect = {"tokens": tokens, "chat_url": settings.gigachat_chat_url}
        return (
            GigaChatClient(model=settings.llm_classify_model, **dialect, **common),
            GigaChatClient(model=settings.llm_generate_model, **dialect, **common),
        )

    raise LLMConfigError(f"неизвестный LLM_PROVIDER: {settings.llm_provider!r}")


def build_classifier_and_generator(settings: Settings) -> tuple[Classifier, ResponseGenerator]:
    if settings.llm_provider == "baseline":
        return BaselineClassifier(), TemplateResponseGenerator()
    classify_client, generate_client = build_llm_clients(settings)
    return LlmClassifier(classify_client), LlmResponseGenerator(generate_client)


@once
def get_agent_service() -> AgentService:
    settings = get_settings()
    classifier, generator = build_classifier_and_generator(settings)

    if settings.simulated_llm_classify_ms or settings.simulated_llm_generate_ms:
        log.warning(
            "ВКЛЮЧЕНА имитация задержки LLM (classify ≤ %d мс, generate ≤ %d мс) - "
            "только для нагрузочного теста",
            settings.simulated_llm_classify_ms,
            settings.simulated_llm_generate_ms,
        )
        classifier = SlowClassifier(classifier, settings.simulated_llm_classify_ms)
        generator = SlowGenerator(generator, settings.simulated_llm_generate_ms)

    graph = TicketGraph(classifier, Retriever(get_embedding_provider()), generator, thresholds())
    return AgentService(graph)
