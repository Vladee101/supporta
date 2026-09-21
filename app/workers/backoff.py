"""Пауза между попытками переподключения воркеров.

Воркер, который падает на первом же сбое Postgres или RabbitMQ, превращает
короткий рестарт инфраструктуры в простой обработки эскалаций до ручного
перезапуска. Поэтому временные сбои пережидаются: пауза растёт экспоненциально
до потолка и сбрасывается после первой успешной операции.

Джиттер разводит по времени несколько экземпляров воркера: иначе после рестарта
брокера они переподключаются синхронно, волной.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class Backoff:
    initial: float = 1.0
    maximum: float = 30.0
    factor: float = 2.0
    jitter: bool = True
    _attempt: int = field(default=0, init=False)

    @property
    def attempt(self) -> int:
        """Число неудачных попыток подряд с последнего `reset()`."""
        return self._attempt

    def next_delay(self) -> float:
        base = min(self.maximum, self.initial * self.factor**self._attempt)
        self._attempt += 1
        if not self.jitter:
            return base
        # «Equal jitter»: не меньше половины паузы, чтобы не долбить сервис.
        return base / 2 + random.uniform(0, base / 2)

    def reset(self) -> None:
        self._attempt = 0
