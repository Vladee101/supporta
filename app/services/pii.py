"""Редакция PII перед вызовом LLM и перед записью в логи (NFR4).

Маскирование стоит первым шагом пайплайна: в модель и в логи уходит
`content_redacted`, оригинал остаётся только в `messages.content` и удаляется
retention-джобом через 90 дней.

Что маскируется и почему именно это: телефон, e-mail, номер карты, паспорт,
СНИЛС, ИНН - прямые идентификаторы, по которым клиента можно найти вне системы.
Номера заказов и трек-номера **не** маскируются: без них ретривер и оператор
теряют предмет обращения, а сами по себе они клиента не идентифицируют.

Имена и адреса регулярками не ловятся без NER и здесь не маскируются - это
осознанная граница, а не недосмотр: см. ограничения в `PiiRedactor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class PiiType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    CARD = "card"
    PASSPORT = "passport"
    SNILS = "snils"
    INN = "inn"


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Текст без PII и статистика находок - статистика идёт в audit_log."""

    text: str
    counts: dict[str, int]

    @property
    def has_pii(self) -> bool:
        return bool(self.counts)


#: Порядок важен: карта проверяется раньше телефона, иначе 16-значный номер
#: карты частично съедается телефонным шаблоном.
_PATTERNS: tuple[tuple[PiiType, re.Pattern[str]], ...] = (
    (PiiType.EMAIL, re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.UNICODE)),
    (PiiType.CARD, re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    (PiiType.SNILS, re.compile(r"\b\d{3}-\d{3}-\d{3}[ -]\d{2}\b")),
    (PiiType.PASSPORT, re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b")),
    (PiiType.INN, re.compile(r"\b\d{12}\b")),
    (
        PiiType.PHONE,
        re.compile(r"(?<!\d)(?:\+7|8)[\s(-]?\d{3}[\s)-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"),
    ),
)


class PiiRedactor:
    """Регулярная редакция PII.

    Ограничения, которые важно знать на приёмке:

    * имена, адреса и названия организаций не маскируются - для них нужен NER,
      он добавляется отдельно и стоит latency (бюджет шага - 0.1 сек);
    * маскирование необратимо и односторонне: восстановить значение из
      `[EMAIL]` нельзя, оператор при необходимости смотрит оригинал тикета;
    * ложные срабатывания предпочтительнее пропусков: 12-значное число
      считается ИНН, даже если это что-то другое.
    """

    def __init__(self, patterns=_PATTERNS) -> None:
        self._patterns = patterns

    def redact(self, text: str) -> RedactionResult:
        counts: dict[str, int] = {}
        redacted = text
        for pii_type, pattern in self._patterns:
            redacted, found = pattern.subn(f"[{pii_type.value.upper()}]", redacted)
            if found:
                counts[pii_type.value] = found
        return RedactionResult(text=redacted, counts=counts)


DEFAULT_REDACTOR = PiiRedactor()


def redact(text: str) -> RedactionResult:
    return DEFAULT_REDACTOR.redact(text)
