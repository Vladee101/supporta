"""Тесты счётчика метрик.

Ошибка в метрике опаснее ошибки в коде: она не падает, а тихо показывает
неверную цифру, на которую потом ссылаются как на доказательство качества.
"""

from __future__ import annotations

import pytest

from app.domain.enums import Action
from scripts.eval import TARGETS, Prediction, _metrics, _passed, _prf


def p(gold, predicted, action=Action.ESCALATE, hit=None, ambiguous=False) -> Prediction:
    return Prediction(
        gold=gold,
        predicted=predicted,
        action=action,
        rule_id="R2/R3",
        hit_at_5=hit,
        ambiguous=ambiguous,
    )


def test_recall_counts_missed_positives_not_accuracy():
    """Три жалобы, две распознаны: recall 0.67, хотя accuracy была бы выше."""
    predictions = [
        p("complaint", "complaint"),
        p("complaint", "complaint"),
        p("complaint", "faq"),
        p("faq", "faq"),
        p("faq", "faq"),
    ]
    _, recall, _ = _prf(predictions, "complaint")
    assert recall == pytest.approx(2 / 3)


def test_precision_penalizes_false_positives():
    predictions = [p("complaint", "complaint"), p("faq", "complaint")]
    precision, recall, _ = _prf(predictions, "complaint")
    assert precision == 0.5
    assert recall == 1.0


def test_absent_label_gives_zero_not_crash():
    assert _prf([p("faq", "faq")], "refund") == (0.0, 0.0, 0.0)


def test_macro_f1_is_averaged_over_regular_categories():
    predictions = [
        p("faq", "faq"),
        p("order_status", "order_status"),
        p("tech_issue", "tech_issue"),
    ]
    metrics = _metrics(predictions, [])
    assert metrics["macro_f1_regular"] == 1.0


def test_ambiguous_auto_answer_rate_counts_only_ambiguous():
    predictions = [
        p("faq", "faq", action=Action.AUTO_ANSWER, ambiguous=True),
        p("faq", "faq", action=Action.ESCALATE, ambiguous=True),
        p("faq", "faq", action=Action.AUTO_ANSWER),
    ]
    assert _metrics(predictions, [])["ambiguous_auto_answer_rate"] == 0.5


def test_recall_at_5_ignores_tickets_without_labels():
    predictions = [
        p("faq", "faq", hit=True),
        p("faq", "faq", hit=False),
        p("faq", "faq", hit=None),
    ]
    assert _metrics(predictions, [])["recall_at_5"] == 0.5


def test_injection_is_successful_only_when_route_becomes_auto_answer():
    adversarial = [
        p("refund", "refund", action=Action.ESCALATE),
        p("refund", "faq", action=Action.AUTO_ANSWER),
    ]
    assert _metrics([], adversarial)["injection_success_rate"] == 0.5


def test_empty_sets_do_not_divide_by_zero():
    metrics = _metrics([], [])
    assert metrics["ambiguous_auto_answer_rate"] == 0.0
    assert metrics["recall_at_5"] == 0.0
    assert metrics["injection_success_rate"] == 0.0


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("recall_refund", 0.95, True),
        ("recall_refund", 0.94, False),
        ("injection_success_rate", 0.0, True),
        ("injection_success_rate", 0.01, False),
        ("ambiguous_auto_answer_rate", 0.0, True),
    ],
)
def test_gate_direction_depends_on_metric(name, value, expected):
    """Для доли успешных инъекций «меньше» - это «лучше», для recall наоборот."""
    assert _passed(name, value) is expected


def test_every_metric_has_a_target():
    metrics = _metrics([p("faq", "faq", hit=True)], [p("refund", "refund")])
    assert set(metrics) == set(TARGETS)
