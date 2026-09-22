"""Консоль показывает оператору подпись, а не сырой код, для каждой причины эскалации.

Регрессия: причина `order_data_unavailable` (R10) появилась в домене позже
консоли, и оператор видел код вместо подписи. Коды - контракт API, поэтому
расхождение ловится тестом на стороне бэкенда.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.domain.enums import Category, EscalationReason

LABELS = Path(__file__).resolve().parent.parent / "console" / "src" / "labels.ts"


def _keys(block_name: str) -> set[str]:
    source = LABELS.read_text(encoding="utf-8")
    block = re.search(rf"{block_name}[^=]*=\s*\{{(.*?)\}};", source, re.S)
    assert block, f"в labels.ts нет {block_name}"
    return set(re.findall(r"^\s*(\w+):", block.group(1), re.M))


def test_every_escalation_reason_has_a_label():
    missing = {reason.value for reason in EscalationReason} - _keys("REASON_LABELS")
    assert not missing, f"нет подписи в console/src/labels.ts: {sorted(missing)}"


def test_every_category_has_a_label():
    missing = {category.value for category in Category} - _keys("CATEGORY_LABELS")
    assert not missing, f"нет подписи в console/src/labels.ts: {sorted(missing)}"
