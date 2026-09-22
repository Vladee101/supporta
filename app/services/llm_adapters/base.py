"""Общее ядро адаптеров LLM-провайдеров (ADR-009).

Здесь всё, что не зависит от формата конкретного API:

* **retry по NFR6**: до `max_retries` повторов с экспоненциальной паузой, но
  не дольше общего дедлайна. Повторяются только временные сбои (сеть, таймаут,
  429, 5xx); 4xx - это ошибка конфигурации, повтор её не исправит;
* **коды категорий**: модель отвечает цифрой 1..N вместо имени метки. Цифра -
  один токен в любом токенизаторе, поэтому распределение по категориям
  читается прямо из `top_logprobs` одной позиции, без склейки вероятностей
  многотокенных меток вроде `order_status`;
* **два способа получить class_confidence** (раздел «Confidence и пороги»):
  logprobs, если провайдер их отдаёт, иначе k-sampling - доля голосов за
  модальную категорию. Режим `auto` пробует logprobs и, если провайдер их не
  вернул, один раз переключается на k-sampling до конца жизни клиента.

Диалекты (OpenAI-совместимый, GigaChat) реализуют только транспорт: два метода
`_label_attempt` и `_text_attempt`.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, TypeVar

import httpx

from app.domain.enums import ConfidenceSource
from app.services.llm import (
    INVALID_LABEL,
    LabelProbabilities,
    LLMUnavailableError,
    ProviderContractError,
)

log = logging.getLogger(__name__)

T = TypeVar("T")

ConfidenceMode = Literal["auto", "logprobs", "k_sampling"]

#: Коды статуса, при которых повтор имеет смысл.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Сколько токенов разрешаем на ответ-классификацию. Одного мало: некоторые
#: модели ставят перед цифрой пробел или перевод строки отдельным токеном.
LABEL_MAX_TOKENS = 4

#: Сколько кандидатов просим на позицию. Должно быть не меньше числа категорий,
#: иначе вероятность части категорий станет нулём не из-за модели, а из-за нас.
TOP_LOGPROBS = 10


class HTTPStatusError(ProviderContractError):
    """Провайдер ответил 4xx, которые не повторяются. Код статуса - для диалектов."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetryableError(Exception):
    """Временный сбой одной попытки: будет повторён, пока позволяет дедлайн."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """NFR6: «retry ×3 (≤ 10 сек), затем эскалация».

    «×3» трактуется как три повтора после первой попытки; дедлайн ограничивает
    всё вместе, включая паузы, - поэтому на практике попыток может быть меньше.
    """

    max_retries: int = 3
    deadline_seconds: float = 10.0
    request_timeout_seconds: float = 8.0
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4.0


@dataclass(frozen=True, slots=True)
class LabelAttempt:
    """Результат одной попытки классификации."""

    text: str
    #: Кандидаты первой позиции, где модель поставила код категории:
    #: список (токен, logprob). None - провайдер logprobs не вернул.
    top_logprobs: list[tuple[str, float]] | None


# ---------------------------------------------------------------------------
# Коды категорий и распределения
# ---------------------------------------------------------------------------


def label_codes(labels: Sequence[str]) -> dict[str, str]:
    """Код → метка: «1» → первая метка и так далее. До 9 категорий."""
    if not 1 <= len(labels) <= 9:
        raise ValueError("кодирование цифрами поддерживает от 1 до 9 категорий")
    return {str(index): label for index, label in enumerate(labels, start=1)}


def codes_instruction(codes: Mapping[str, str]) -> str:
    """Добавка к системному промпту: как отвечать кодом, а не именем категории."""
    mapping = "\n".join(f"{code} - {label}" for code, label in codes.items())
    return f"Ответь ровно одной цифрой - кодом категории, без пояснений:\n{mapping}"


def parse_code(text: str, codes: Mapping[str, str]) -> str | None:
    """Первый непробельный символ ответа, если это допустимый код."""
    stripped = text.strip()
    return stripped[0] if stripped and stripped[0] in codes else None


def distribution_from_logprobs(
    top: Sequence[tuple[str, float]], codes: Mapping[str, str]
) -> dict[str, float] | None:
    """Нормированное распределение по категориям из кандидатов одной позиции.

    Вероятности нормируются среди допустимых кодов - это и есть «нормализованная
    вероятность выбранной метки среди допустимых» из раздела «Confidence и
    пороги». Коды, не попавшие в top-k, получают 0. Если ни одного кода среди
    кандидатов нет - распределения нет (None), а не равномерная догадка.
    """
    best: dict[str, float] = {}
    for token, logprob in top:
        code = token.strip()
        if code in codes and (code not in best or logprob > best[code]):
            best[code] = logprob
    if not best:
        return None

    peak = max(best.values())
    weights = {code: math.exp(logprob - peak) for code, logprob in best.items()}
    total = sum(weights.values())
    return {codes[code]: weight / total for code, weight in weights.items()}


def distribution_from_votes(answers: Sequence[str], codes: Mapping[str, str]) -> dict[str, float]:
    """k-sampling: доля голосов. Нераспознанный ответ - голос «ни за кого».

    Знаменатель - все k ответов, включая нераспознанные: мусор в ответах
    должен снижать уверенность, а не исчезать из статистики.
    """
    votes: dict[str, int] = {}
    for answer in answers:
        code = parse_code(answer, codes)
        if code is not None:
            votes[codes[code]] = votes.get(codes[code], 0) + 1
    if not votes:
        return {INVALID_LABEL: 1.0}
    return {label: count / len(answers) for label, count in votes.items()}


# ---------------------------------------------------------------------------
# Базовый клиент
# ---------------------------------------------------------------------------


class BaseLLMClient:
    """Классификация и генерация поверх двух транспортных методов диалекта."""

    def __init__(
        self,
        *,
        model: str,
        http: httpx.Client,
        retry: RetryPolicy | None = None,
        confidence_mode: ConfidenceMode = "auto",
        k_samples: int = 5,
        sampling_temperature: float | None = 0.7,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if k_samples < 1:
            raise ValueError("k_samples должен быть ≥ 1")
        self.model_id = model
        self._model = model
        self._http = http
        self._retry = retry or RetryPolicy()
        self._mode = confidence_mode
        self._k = k_samples
        self._temperature = sampling_temperature
        self._sleep = sleep
        self._clock = clock
        # auto: после первого ответа без logprobs больше их не просим.
        self._logprobs_unavailable = confidence_mode == "k_sampling"
        self._mode_lock = threading.Lock()

    # --- контракт диалекта --------------------------------------------------

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
        raise NotImplementedError

    def _text_attempt(self, system: str, user: str, *, max_tokens: int, timeout: float) -> str:
        raise NotImplementedError

    # --- retry ----------------------------------------------------------------

    def _with_retry(self, operation: Callable[[float], T], *, deadline: float) -> T:
        """Выполнить `operation(timeout)` с повторами временных сбоев до дедлайна."""
        policy = self._retry
        last_error: Exception | None = None

        for attempt in range(policy.max_retries + 1):
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            try:
                return operation(min(policy.request_timeout_seconds, remaining))
            except RetryableError as exc:
                last_error = exc
                if attempt == policy.max_retries:
                    break
                pause = exc.retry_after
                if pause is None:
                    backoff = policy.base_backoff_seconds * (2**attempt)
                    # Джиттер: одновременно упавшие запросы не должны
                    # вернуться к провайдеру одной волной.
                    pause = min(policy.max_backoff_seconds, backoff) * random.uniform(0.8, 1.2)
                if self._clock() + pause >= deadline:
                    break  # следующая попытка всё равно не уложится в дедлайн
                self._sleep(pause)

        raise LLMUnavailableError(
            f"{self.model_id}: провайдер недоступен в пределах "
            f"{policy.deadline_seconds:.0f} с: {last_error}"
        ) from last_error

    def _deadline(self) -> float:
        return self._clock() + self._retry.deadline_seconds

    # --- классификация --------------------------------------------------------

    def classify(self, system: str, text: str, labels: Sequence[str]) -> LabelProbabilities:
        codes = label_codes(labels)
        full_system = f"{system}\n\n{codes_instruction(codes)}"
        deadline = self._deadline()

        if not self._logprobs_unavailable:
            attempt = self._with_retry(
                lambda timeout: self._label_attempt(
                    full_system, text, codes, want_logprobs=True, temperature=None, timeout=timeout
                ),
                deadline=deadline,
            )
            if attempt.top_logprobs is not None:
                distribution = distribution_from_logprobs(attempt.top_logprobs, codes)
                if distribution is None:
                    # logprobs есть, но кода среди кандидатов нет: модель ответила
                    # не цифрой. Уверенности нет - это R9, а не догадка.
                    distribution = {INVALID_LABEL: 1.0}
                return LabelProbabilities(
                    probabilities=distribution,
                    source=ConfidenceSource.LOGPROBS,
                    model_id=self.model_id,
                    reasoning=f"ответ модели: {attempt.text.strip()[:20]!r}",
                )
            self._on_logprobs_missing()

        return self._classify_by_sampling(full_system, text, codes, deadline)

    def _on_logprobs_missing(self) -> None:
        if self._mode == "logprobs":
            raise ProviderContractError(
                f"{self.model_id}: провайдер не вернул logprobs, а режим требует их "
                "(LLM_CONFIDENCE_MODE=logprobs). Для провайдеров без logprobs - auto или k_sampling"
            )
        with self._mode_lock:
            if not self._logprobs_unavailable:
                log.warning(
                    "%s: провайдер не отдаёт logprobs - class_confidence считается "
                    "k-sampling (k=%d), стоимость классификации ×%d",
                    self.model_id,
                    self._k,
                    self._k,
                )
                self._logprobs_unavailable = True

    def _classify_by_sampling(
        self, system: str, text: str, codes: Mapping[str, str], deadline: float
    ) -> LabelProbabilities:
        # Одна выборка - это не голосование: нужен самый вероятный ответ, а не
        # случайный. Температура для разброса голосов имеет смысл только при k > 1.
        temperature = self._temperature
        if temperature is not None and self._k == 1:
            temperature = 0.0

        def one_sample() -> str | None:
            try:
                attempt = self._with_retry(
                    lambda timeout: self._label_attempt(
                        system,
                        text,
                        codes,
                        want_logprobs=False,
                        temperature=temperature,
                        timeout=timeout,
                    ),
                    deadline=deadline,
                )
            except ProviderContractError:
                raise
            except LLMUnavailableError:
                return None
            return attempt.text

        # Выборки параллельно: задержка ≈ одному вызову, а не k (бюджет NFR1).
        with ThreadPoolExecutor(max_workers=self._k) as pool:
            results = list(pool.map(lambda _: one_sample(), range(self._k)))

        answers = [answer for answer in results if answer is not None]
        if not answers:
            raise LLMUnavailableError(f"{self.model_id}: ни одна из {self._k} выборок не удалась")
        # Упавшая выборка считается как нераспознанный голос: неполные данные
        # должны снижать уверенность, а не незаметно сокращать k.
        answers += [""] * (self._k - len(answers))

        return LabelProbabilities(
            probabilities=distribution_from_votes(answers, codes),
            source=ConfidenceSource.K_SAMPLING,
            model_id=self.model_id,
            reasoning=f"голоса: {[a.strip()[:3] for a in answers]}",
        )

    # --- генерация -------------------------------------------------------------

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        text = self._with_retry(
            lambda timeout: self._text_attempt(
                system, user, max_tokens=max_tokens, timeout=timeout
            ),
            deadline=self._deadline(),
        )
        if not text.strip():
            raise ProviderContractError(f"{self.model_id}: пустой ответ генерации")
        return text


# ---------------------------------------------------------------------------
# HTTP-вспомогательное для диалектов
# ---------------------------------------------------------------------------


def post_json(
    http: httpx.Client, url: str, *, json: dict, headers: dict[str, str], timeout: float
) -> httpx.Response:
    """POST с разбором ошибок на временные (повторяемые) и контрактные."""
    try:
        response = http.post(url, json=json, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise RetryableError(f"таймаут: {exc!r}") from exc
    except httpx.TransportError as exc:
        raise RetryableError(f"сетевая ошибка: {exc!r}") from exc

    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code}", retry_after=_retry_after(response))
    if response.status_code >= 400:
        raise HTTPStatusError(
            response.status_code, f"HTTP {response.status_code}: {response.text[:300]}"
        )
    return response


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None
