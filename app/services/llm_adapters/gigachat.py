"""Диалект GigaChat (Сбер): собственный контракт chat completions v2 и OAuth.

Отличия от OpenAI, из-за которых нужен отдельный диалект (сверено с официальным
клиентом ai-forever/gigachat):

* аутентификация - OAuth client credentials: ключ авторизации (base64) меняется
  на access token сроком 30 минут; запрос требует заголовок `RqUID` и `scope`;
* `POST /v2/chat/completions`, параметры генерации вложены в `model_options`;
* содержимое сообщения - список частей `[{"text": ...}]`;
* logprobs - `messages[].logprobs = [{"chosen": {...}, "top": [{token, logprob}]}]`;
* ограничение ответа регулярным выражением: `response_format = {type: regex}` -
  им ответ сужается до кодов категорий.

Сертификаты: API GigaChat подписан Russian Trusted Root CA. Путь к бандлу -
настройка `LLM_CA_BUNDLE`; отключать проверку TLS нельзя.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from app.services import llm_usage
from app.services.llm import LLMUnavailableError, ProviderContractError
from app.services.llm_adapters.base import (
    LABEL_MAX_TOKENS,
    TOP_LOGPROBS,
    BaseLLMClient,
    HTTPStatusError,
    LabelAttempt,
    RetryableError,
    post_json,
)

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://api.giga.chat/v2/chat/completions"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"

#: Обновляем токен за минуту до истечения - как официальный клиент.
EXPIRY_BUFFER_MS = 60_000


class GigaChatTokenProvider:
    """Кеширует access token и обновляет его заранее. Потокобезопасен.

    Один экземпляр на процесс - общий для классификатора и генератора: иначе
    каждый получал бы свой токен, а лимит на выдачу токенов у провайдера есть.
    """

    def __init__(
        self,
        *,
        credentials: str,
        http: httpx.Client,
        scope: str = DEFAULT_SCOPE,
        auth_url: str = AUTH_URL,
        now_ms: Callable[[], float] = lambda: time.time() * 1000,
    ) -> None:
        self._credentials = credentials
        self._http = http
        self._scope = scope
        self._auth_url = auth_url
        self._now_ms = now_ms
        self._token: str | None = None
        self._expires_at_ms = 0.0
        self._lock = threading.Lock()

    def token(self, *, timeout: float, force_refresh: bool = False) -> str:
        with self._lock:
            fresh = self._token and self._expires_at_ms > self._now_ms() + EXPIRY_BUFFER_MS
            if fresh and not force_refresh:
                return self._token
            self._token, self._expires_at_ms = self._fetch(timeout)
            return self._token

    def _fetch(self, timeout: float) -> tuple[str, float]:
        try:
            response = self._http.post(
                self._auth_url,
                data={"scope": self._scope},
                headers={
                    "Authorization": f"Basic {self._credentials}",
                    "RqUID": str(uuid.uuid4()),
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"OAuth GigaChat недоступен: {exc!r}") from exc

        if response.status_code >= 500 or response.status_code == 429:
            raise RetryableError(f"OAuth GigaChat: HTTP {response.status_code}")
        if response.status_code != 200:
            raise ProviderContractError(
                f"OAuth GigaChat: HTTP {response.status_code} - проверьте ключ авторизации и scope"
            )
        payload = response.json()
        # Официальный клиент принимает обе формы ответа: access_token/expires_at и tok/exp.
        token = payload.get("access_token") or payload.get("tok")
        expires_at = payload.get("expires_at") or payload.get("exp")
        if not token or not expires_at:
            raise ProviderContractError("OAuth GigaChat: в ответе нет токена или срока действия")
        return token, float(expires_at)


class GigaChatClient(BaseLLMClient):
    def __init__(
        self,
        *,
        tokens: GigaChatTokenProvider,
        chat_url: str = CHAT_URL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tokens = tokens
        self._chat_url = chat_url

    def _call(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        for refreshed in (False, True):
            token = self._tokens.token(timeout=timeout, force_refresh=refreshed)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            try:
                response = post_json(
                    self._http, self._chat_url, json=body, headers=headers, timeout=timeout
                )
            except HTTPStatusError as exc:
                # 401 на чате - токен отозван раньше срока: один раз берём новый.
                if exc.status_code == 401 and not refreshed:
                    continue
                raise
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderContractError(
                    f"ответ GigaChat не JSON: {response.text[:300]}"
                ) from exc
            # Вызов оплачен, даже если дальше ответ не пройдёт проверку контракта.
            usage = payload.get("usage") if isinstance(payload, dict) else None
            llm_usage.record(self._model, usage)
            return payload
        raise LLMUnavailableError("GigaChat отклонил обновлённый токен")

    def _body(self, system: str, user: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": [{"text": system}]},
                {"role": "user", "content": [{"text": user}]},
            ],
            "model_options": options,
        }

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
        options: dict[str, Any] = {
            "max_tokens": LABEL_MAX_TOKENS,
            # Декодер физически не может ответить ничем, кроме кода категории.
            "response_format": {"type": "regex", "regex": f"[{''.join(codes)}]"},
        }
        if want_logprobs:
            options["top_logprobs"] = TOP_LOGPROBS
        if temperature is not None:
            options["temperature"] = temperature

        payload = self._call(self._body(system, text, options), timeout)
        message = _last_assistant_message(payload)
        return LabelAttempt(
            text=_text_of(message),
            top_logprobs=_code_position_logprobs(message, payload, codes),
        )

    def _text_attempt(self, system: str, user: str, *, max_tokens: int, timeout: float) -> str:
        payload = self._call(self._body(system, user, {"max_tokens": max_tokens}), timeout)
        return _text_of(_last_assistant_message(payload))


def _last_assistant_message(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProviderContractError(f"ответ GigaChat без messages: {str(payload)[:300]}")
    return messages[-1]


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content") or []
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content if isinstance(part, dict))


def _code_position_logprobs(
    message: dict[str, Any], payload: dict[str, Any], codes: Mapping[str, str]
) -> list[tuple[str, float]] | None:
    positions = message.get("logprobs") or payload.get("logprobs")
    if not positions:
        return None

    def candidates(position: dict[str, Any]) -> list[tuple[str, float]]:
        items = list(position.get("top") or [])
        if position.get("chosen"):
            items.append(position["chosen"])
        return [(item["token"], float(item["logprob"])) for item in items]

    for position in positions:
        chosen = (position.get("chosen") or {}).get("token", "")
        if chosen.strip() in codes:
            return candidates(position)
    return candidates(positions[0])
