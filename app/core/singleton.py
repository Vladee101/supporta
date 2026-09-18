"""Потокобезопасная ленивая инициализация тяжёлых объектов процесса.

`functools.lru_cache` не держит блокировку на время вызова функции: если N
потоков одновременно промахиваются по пустому кешу, функция выполняется N раз.
Нагрузочный тест поймал это на `get_engine()`: первая волна из 50 запросов
создала до 50 движков SQLAlchemy - каждый со своим пулом, - и число соединений
с Postgres перестало быть ограниченным. Для `get_embedding_provider()` то же
самое означало бы 50 параллельных загрузок bge-m3 по 2 ГБ.

`@once` гарантирует ровно один вызов (double-checked locking) и сохраняет
`cache_clear()`, на который опираются тесты.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_UNSET = object()


def once(factory: Callable[[], T]) -> Callable[[], T]:
    lock = threading.Lock()
    value: object = _UNSET

    @functools.wraps(factory)
    def wrapper() -> T:
        nonlocal value
        if value is _UNSET:  # быстрый путь без блокировки, когда значение уже есть
            with lock:
                if value is _UNSET:
                    value = factory()
        return value  # type: ignore[return-value]

    def cache_clear() -> None:
        nonlocal value
        with lock:
            value = _UNSET

    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    return wrapper
