"""@once: ровно одна инициализация при одновременном первом обращении."""

from __future__ import annotations

import threading
import time

import pytest

from app.core.singleton import once


def test_concurrent_first_calls_initialize_once():
    """Регрессия: lru_cache создавал по движку SQLAlchemy на каждый поток первой волны."""
    calls = []

    @once
    def factory():
        calls.append(1)
        time.sleep(0.05)  # окно гонки: пока один поток строит объект, остальные ждут
        return object()

    barrier = threading.Barrier(20)
    results = []

    def hit():
        barrier.wait()
        results.append(factory())

    threads = [threading.Thread(target=hit) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert len({id(value) for value in results}) == 1


def test_cache_clear_allows_reinitialization():
    counter = iter(range(10))

    @once
    def factory():
        return next(counter)

    assert factory() == 0
    assert factory() == 0
    factory.cache_clear()
    assert factory() == 1


def test_failed_initialization_is_retried():
    """Исключение не кешируется: следующий вызов пробует снова."""
    attempts = []

    @once
    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("broker down")
        return "ok"

    with pytest.raises(RuntimeError):
        factory()
    assert factory() == "ok"
