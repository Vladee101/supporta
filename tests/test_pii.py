"""Тесты PII-редакции (NFR4)."""

from __future__ import annotations

import pytest

from app.services.pii import PiiType, redact


@pytest.mark.parametrize(
    ("text", "pii_type"),
    [
        ("напишите на ivan.petrov+shop@example.com", PiiType.EMAIL),
        ("мой телефон +7 999 123-45-67", PiiType.PHONE),
        ("звоните 8(999)1234567", PiiType.PHONE),
        ("карта 4276 3800 1234 5678", PiiType.CARD),
        ("паспорт 45 12 345678", PiiType.PASSPORT),
        ("снилс 112-233-445 95", PiiType.SNILS),
        ("инн 771234567890", PiiType.INN),
    ],
)
def test_direct_identifiers_are_masked(text, pii_type):
    result = redact(text)
    assert f"[{pii_type.value.upper()}]" in result.text
    assert result.counts[pii_type.value] == 1


def test_original_value_does_not_survive_redaction():
    result = redact("почта client@mail.ru, телефон +79991234567")
    assert "client@mail.ru" not in result.text
    assert "79991234567" not in result.text
    assert result.has_pii


@pytest.mark.parametrize(
    "text",
    [
        "заказ №1023456 не пришёл",
        "трек-номер RU123456789CN",
        "ошибка ERR-5012 при оплате",
    ],
)
def test_order_and_tracking_numbers_are_not_masked(text):
    """Их маскирование лишило бы ретривер и оператора предмета обращения."""
    assert redact(text).text == text


def test_card_is_matched_before_phone():
    """Порядок шаблонов: иначе телефонная маска съедает часть номера карты."""
    result = redact("оплатил картой 4276380012345678")
    assert result.counts == {PiiType.CARD.value: 1}


def test_clean_text_is_untouched():
    text = "Здравствуйте! Когда приедет мой заказ?"
    result = redact(text)
    assert result.text == text
    assert not result.has_pii
    assert result.counts == {}


def test_counts_are_reported_per_type():
    result = redact("a@b.ru и c@d.ru, телефон +7 999 111-22-33")
    assert result.counts == {PiiType.EMAIL.value: 2, PiiType.PHONE.value: 1}
