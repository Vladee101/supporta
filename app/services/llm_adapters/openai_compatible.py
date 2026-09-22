"""Диалект OpenAI Chat Completions: YandexGPT (AI Studio), vLLM, любой совместимый API.

Примеры конфигурации:

* YandexGPT: `base_url=https://llm.api.cloud.yandex.net/v1`,
  модель `gpt://<folder_id>/yandexgpt/latest`, ключ API в `Authorization: Bearer`;
* vLLM: `base_url=http://gpu-host:8000/v1`, модель - имя загруженных весов,
  `choice_constraint=guided_choice` (или `structured_outputs` для новых версий
  vLLM) - ответ ограничивается кодами категорий на уровне декодера.

Какие параметры провайдер реально поддерживает, адаптер не угадывает: logprobs
запрашиваются, и если их нет в ответе, режим `auto` переходит на k-sampling.
Параметр `n` не используется - многие совместимые API принимают только `n=1`,
поэтому k выборок - это k параллельных запросов.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import httpx

from app.services import llm_usage
from app.services.llm import ProviderContractError
from app.services.llm_adapters.base import (
    LABEL_MAX_TOKENS,
    TOP_LOGPROBS,
    BaseLLMClient,
    LabelAttempt,
    post_json,
)

ChoiceConstraint = Literal["none", "guided_choice", "structured_outputs"]


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        auth_scheme: str = "Bearer",
        choice_constraint: ChoiceConstraint = "none",
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            self._headers["Authorization"] = f"{auth_scheme} {api_key}"
        self._constraint = choice_constraint

    def _body(self, system: str, user: str, *, max_tokens: int) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }

    def _post(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        response = post_json(
            self._http, self._url, json=body, headers=self._headers, timeout=timeout
        )
        try:
            payload = response.json()
            # Вызов оплачен, даже если дальше ответ не пройдёт проверку контракта.
            llm_usage.record(self._model, payload.get("usage"))
            choice = payload["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderContractError(
                f"ответ не в формате chat completions: {response.text[:300]}"
            ) from exc
        return choice

    def _label_attempt(
        self,
        system: str,
        text: str,
        codes: Mapping[str, str],
        *,
        want_logprobs: bool,
        temperature: float | None,
        timeout: float,
    ) -> LabelAttempt:
        body = self._body(system, text, max_tokens=LABEL_MAX_TOKENS)
        if temperature is not None:
            body["temperature"] = temperature
        if want_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = TOP_LOGPROBS
        if self._constraint == "guided_choice":
            body["guided_choice"] = list(codes)
        elif self._constraint == "structured_outputs":
            body["structured_outputs"] = {"choice": list(codes)}

        choice = self._post(body, timeout)
        content = (choice.get("message") or {}).get("content") or ""
        return LabelAttempt(text=content, top_logprobs=_code_position_logprobs(choice, codes))

    def _text_attempt(self, system: str, user: str, *, max_tokens: int, timeout: float) -> str:
        choice = self._post(self._body(system, user, max_tokens=max_tokens), timeout)
        return (choice.get("message") or {}).get("content") or ""


def _code_position_logprobs(
    choice: dict[str, Any], codes: Mapping[str, str]
) -> list[tuple[str, float]] | None:
    """Кандидаты первой позиции, где выбранный токен - код категории.

    Первая позиция не всегда годится: модель может начать ответ с пробела или
    перевода строки отдельным токеном. Если кода нет ни в одной позиции, берём
    первую - distribution_from_logprobs вернёт None, и классификация уйдёт в R9.
    """
    logprobs = choice.get("logprobs")
    if not logprobs or not logprobs.get("content"):
        return None

    positions = logprobs["content"]
    for position in positions:
        if position.get("token", "").strip() in codes:
            return _candidates(position)
    return _candidates(positions[0])


def _candidates(position: dict[str, Any]) -> list[tuple[str, float]]:
    top = position.get("top_logprobs") or []
    candidates = [(item["token"], float(item["logprob"])) for item in top]
    # Выбранный токен тоже кандидат: некоторые реализации не дублируют его в top.
    if "token" in position and "logprob" in position:
        candidates.append((position["token"], float(position["logprob"])))
    return candidates


def build_http_client(verify: bool | str = True) -> httpx.Client:
    """Один пул соединений на процесс: TLS-рукопожатие на каждый тикет - лишние 100+ мс."""
    return httpx.Client(
        verify=verify,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
