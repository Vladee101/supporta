"""Учёт расхода LLM на тикет (NFR5, SLI «стоимость на тикет»).

Адаптеры не знают, какой тикет обрабатывают, а агент не знает, сколько вызовов
сделал адаптер: при k-sampling и повторах по NFR6 их может быть несколько на
один шаг графа. Поэтому расход собирается через контекст: агент открывает
счётчик на время обработки тикета, адаптеры пишут в текущий счётчик `usage` из
каждого ответа провайдера. Вне счётчика запись - no-op.

Стоимость берётся из ответа провайдера, если он её сообщает (RouterAI отдаёт
`usage.cost` в рублях). Если нет - стоимость неизвестна, и это видно в итоге:
`calls_without_cost` > 0. Досчитывать её по прайсу здесь не нужно - прайс
меняется, а трейс должен хранить факт, а не оценку.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    calls_without_cost: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: Mapping[str, Any] | None) -> None:
        usage = usage or {}
        cost = usage.get("cost")
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            if isinstance(cost, int | float):
                self.cost += float(cost)
            else:
                self.calls_without_cost += 1
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                # None, а не 0: неизвестная стоимость не должна выглядеть бесплатной.
                "cost": round(self.cost, 6) if self.calls_without_cost == 0 else None,
                "calls_without_cost": self.calls_without_cost,
                "by_model": dict(self.by_model),
            }


_current: ContextVar[UsageMeter | None] = ContextVar("llm_usage_meter", default=None)


@contextmanager
def metered() -> Iterator[UsageMeter]:
    """Счётчик расхода на время обработки одного тикета."""
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def record(model: str, usage: Mapping[str, Any] | None) -> None:
    """Записать расход одного вызова в текущий счётчик, если он открыт."""
    meter = _current.get()
    if meter is not None:
        meter.add(model, usage)
