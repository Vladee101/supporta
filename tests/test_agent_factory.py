"""Фабрика агента: выбор провайдера по конфигурации (ADR-009)."""

from __future__ import annotations

import pytest

from app.agent.factory import LLMConfigError, build_classifier_and_generator
from app.core.config import Settings
from app.services.classifier import BaselineClassifier, LlmClassifier
from app.services.generation import LlmResponseGenerator, TemplateResponseGenerator
from app.services.llm_adapters import GigaChatClient, OpenAICompatibleClient


def settings(**overrides) -> Settings:
    # _env_file=None: тест не должен зависеть от .env разработчика.
    return Settings(_env_file=None, **overrides)


def test_baseline_needs_no_network_or_keys():
    classifier, generator = build_classifier_and_generator(settings())
    assert isinstance(classifier, BaselineClassifier)
    assert isinstance(generator, TemplateResponseGenerator)


def test_openai_compatible_builds_two_models():
    classifier, generator = build_classifier_and_generator(
        settings(
            llm_provider="openai_compatible",
            llm_base_url="https://llm.api.cloud.yandex.net/v1",
            llm_api_key="key",
            llm_classify_model="gpt://folder/yandexgpt-lite/latest",
            llm_generate_model="gpt://folder/yandexgpt/latest",
        )
    )
    assert isinstance(classifier, LlmClassifier) and isinstance(generator, LlmResponseGenerator)
    assert classifier.model_id == "gpt://folder/yandexgpt-lite/latest"
    assert generator.model_id == "gpt://folder/yandexgpt/latest"
    assert isinstance(classifier._client, OpenAICompatibleClient)


def test_gigachat_shares_one_token_provider():
    """Один токен на процесс: лимит на выдачу токенов у провайдера есть."""
    classifier, generator = build_classifier_and_generator(
        settings(
            llm_provider="gigachat",
            gigachat_credentials="YmFzZTY0",
            llm_classify_model="GigaChat-2",
            llm_generate_model="GigaChat-2-Pro",
        )
    )
    assert isinstance(classifier._client, GigaChatClient)
    assert classifier._client._tokens is generator._client._tokens


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_provider": "openai_compatible", "llm_classify_model": "a", "llm_generate_model": "b"},
        {"llm_provider": "gigachat", "llm_classify_model": "a", "llm_generate_model": "b"},
        {"llm_provider": "gigachat", "gigachat_credentials": "x"},
    ],
)
def test_incomplete_provider_config_fails_at_startup(overrides):
    """Неполная настройка - ошибка при старте, а не эскалация каждого тикета."""
    with pytest.raises(LLMConfigError):
        build_classifier_and_generator(settings(**overrides))
