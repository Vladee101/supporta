"""Контракт LLM-провайдера.

Домен и пайплайн зависят от этого протокола, а не от конкретного вендора:
ADR-009 требует возможности менять провайдера, не трогая Decision Engine.

Два требования к провайдеру, из которых следует всё остальное:

* **structured output** - категория должна приходить как значение из закрытого
  множества меток, а не как свободный текст, который нужно парсить;
* **logprobs** - `class_confidence` берётся из вероятности выбранной метки.
  Если провайдер их не отдаёт, реализация обязана вернуть
  `ConfidenceSource.K_SAMPLING` и посчитать долю голосов; число, которое модель
  назвала «своей уверенностью» в тексте ответа, использовать нельзя - оно не
  калибровано и систематически завышено (см. «Confidence и пороги»).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.domain.enums import ConfidenceSource


class LLMUnavailableError(RuntimeError):
    """Провайдер недоступен после всех retry - ведёт к эскалации (NFR6)."""


class ProviderContractError(LLMUnavailableError):
    """Провайдер ответил не по контракту: 4xx, неожиданный формат, нет обещанных logprobs.

    Повтор не поможет - это ошибка конфигурации или несовместимость API, а не
    временный сбой. Для пайплайна исход тот же - эскалация, - но в логе она
    должна отличаться от сетевого сбоя.
    """


#: Метка, которую адаптер возвращает, когда ни один ответ модели не распознан
#: как допустимая категория. Её нет в закрытом множестве, поэтому LlmClassifier
#: превращает её в UNCLASSIFIED - и Decision Engine эскалирует по R9.
INVALID_LABEL = "__invalid__"


@dataclass(frozen=True, slots=True)
class LabelProbabilities:
    """Распределение по допустимым меткам плюс способ его получения."""

    probabilities: Mapping[str, float]
    source: ConfidenceSource
    model_id: str
    reasoning: str | None = None

    def top(self) -> tuple[str, float]:
        label = max(self.probabilities, key=lambda key: self.probabilities[key])
        return label, self.probabilities[label]


@runtime_checkable
class LLMClient(Protocol):
    """Минимальный интерфейс провайдера, которого достаточно пайплайну."""

    model_id: str

    def classify(self, system: str, text: str, labels: Sequence[str]) -> LabelProbabilities:
        """Вернуть распределение по меткам.

        `system` описывает смысл меток, `text` - обращение, обёрнутое как данные.
        Инструкции внутри `text` не выполняются (NFR10).
        """
        ...

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        """Сгенерировать ответ. Используется только для черновиков и автоответов."""
        ...
