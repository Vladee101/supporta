"""Тесты Decision Engine: строки таблицы, инварианты, полный перебор входов.

Соответствие design document → «Стратегия тестирования», уровень
«Unit: Decision Engine»: 100% строк таблицы, полный перебор 144 комбинаций,
ни одного попадания в R-default, 5 инвариантов.
"""

from __future__ import annotations

import itertools

import pytest

from app.domain.decision import (
    PRIORITY_CLIENT_REQUESTED,
    PRIORITY_STANDARD,
    RULES,
    DecisionInput,
    Thresholds,
    decide,
)
from app.domain.enums import (
    HIGH_RISK_CATEGORIES,
    Action,
    Category,
    EscalationReason,
)

T = Thresholds()


def ti(
    category: Category,
    *,
    rag: float | None = None,
    cls: float | None = None,
    iteration: int = 0,
    human: bool = False,
) -> DecisionInput:
    """Короткий конструктор входа для тестов (порядок confidence - только по имени)."""
    return DecisionInput(
        category,
        rag_confidence=rag,
        class_confidence=cls,
        clarification_iteration=iteration,
        human_requested=human,
    )


# Бакеты входного пространства. Ровно те, что посчитаны в design document:
# 6 категорий × 3 диапазона RAG × 2 диапазона class × 2 итерации × 2 флага = 144.
CATEGORIES = tuple(Category)
RAG_BUCKETS = (None, 0.4, 0.9)  # нет измерения / ниже порога / выше порога
CLASS_BUCKETS = (0.5, 0.95)  # ниже порога / выше порога
ITERATIONS = (0, 1)
HUMAN_FLAGS = (False, True)

ALL_INPUTS = [
    ti(category, rag=rag, cls=cls, iteration=iteration, human=human)
    for category, rag, cls, iteration, human in itertools.product(
        CATEGORIES, RAG_BUCKETS, CLASS_BUCKETS, ITERATIONS, HUMAN_FLAGS
    )
]


def test_input_space_size_matches_design_document():
    assert len(ALL_INPUTS) == 144


# --------------------------------------------------------------------------
# Строки decision table: по одному каноническому случаю на каждое правило
# --------------------------------------------------------------------------

ROW_CASES = [
    pytest.param(
        ti(Category.COMPLAINT, rag=0.95, cls=0.9, human=True),
        "R1",
        Action.PRIORITY_ESCALATE,
        EscalationReason.CLIENT_REQUESTED,
        id="R1-запрос человека бьёт high-risk категорию",
    ),
    pytest.param(
        ti(Category.UNCLASSIFIED),
        "R9",
        Action.ESCALATE,
        EscalationReason.CLASSIFICATION_FAILED,
        id="R9-классификация не удалась",
    ),
    pytest.param(
        ti(Category.COMPLAINT, rag=0.99, cls=0.99),
        "R2/R3",
        Action.ESCALATE,
        EscalationReason.HIGH_RISK_CATEGORY,
        id="R2/R3-жалоба при максимальном confidence",
    ),
    pytest.param(
        ti(Category.REFUND, rag=0.2, cls=0.2),
        "R2/R3",
        Action.ESCALATE,
        EscalationReason.HIGH_RISK_CATEGORY,
        id="R2/R3-возврат при низком confidence",
    ),
    pytest.param(
        ti(Category.TECH_ISSUE, rag=0.9, cls=0.9, iteration=1),
        "R7b",
        Action.ESCALATE,
        EscalationReason.CLARIFICATION_LIMIT_REACHED,
        id="R7b-лимит уточнений исчерпан",
    ),
    pytest.param(
        ti(Category.TECH_ISSUE, rag=0.95, cls=0.5),
        "R7c",
        Action.ESCALATE,
        EscalationReason.LOW_CLASS_CONFIDENCE,
        id="R7c-дыра, закрытая в ревизии 2",
    ),
    pytest.param(
        ti(Category.TECH_ISSUE, rag=0.95, cls=0.9),
        "R7a",
        Action.CLARIFY,
        None,
        id="R7a-запрос уточнения",
    ),
    pytest.param(
        ti(Category.TECH_ISSUE, rag=0.3, cls=0.9),
        "R8",
        Action.ESCALATE,
        EscalationReason.LOW_RAG_CONFIDENCE,
        id="R8-техпроблема без опоры в KB",
    ),
    pytest.param(
        ti(Category.FAQ, rag=0.9, cls=0.5),
        "R5",
        Action.ESCALATE,
        EscalationReason.LOW_CLASS_CONFIDENCE,
        id="R5-FAQ с низким class confidence",
    ),
    pytest.param(
        ti(Category.FAQ, rag=0.3, cls=0.95),
        "R6",
        Action.ESCALATE,
        EscalationReason.LOW_RAG_CONFIDENCE,
        id="R6-FAQ без опоры в KB",
    ),
    pytest.param(
        ti(Category.ORDER_STATUS, rag=0.99, cls=0.99),
        "R10",
        Action.ESCALATE,
        EscalationReason.ORDER_DATA_UNAVAILABLE,
        id="R10-статус заказа на оператора даже при максимальном confidence",
    ),
    pytest.param(
        ti(Category.FAQ, rag=0.9, cls=0.95),
        "R4",
        Action.AUTO_ANSWER,
        None,
        id="R4-единственный путь к автоответу",
    ),
]


@pytest.mark.parametrize(("decision_input", "rule_id", "action", "reason"), ROW_CASES)
def test_decision_table_rows(decision_input, rule_id, action, reason):
    decision = decide(decision_input, T)
    assert decision.rule_id == rule_id
    assert decision.action == action
    assert decision.reason == reason


def test_every_rule_is_reachable():
    """Недостижимое правило - такой же дефект таблицы, как и непокрытая комбинация."""
    fired = {decide(i, T).rule_id for i in ALL_INPUTS}
    declared = {rule.rule_id for rule in RULES}
    assert declared - fired == set(), f"недостижимые правила: {declared - fired}"


# --------------------------------------------------------------------------
# Инварианты 1-5 из design document
# --------------------------------------------------------------------------


def test_invariant_1_human_request_always_wins():
    for decision_input in ALL_INPUTS:
        if decision_input.human_requested:
            decision = decide(decision_input, T)
            assert decision.rule_id == "R1", decision_input
            assert decision.action is Action.PRIORITY_ESCALATE
            assert decision.priority == PRIORITY_CLIENT_REQUESTED


def test_invariant_2_high_risk_categories_never_auto_answered():
    for decision_input in ALL_INPUTS:
        if decision_input.category in HIGH_RISK_CATEGORIES:
            assert decide(decision_input, T).action is not Action.AUTO_ANSWER, decision_input


def test_invariant_3_low_class_confidence_never_auto_answered():
    for decision_input in ALL_INPUTS:
        low = (
            decision_input.class_confidence is None
            or decision_input.class_confidence < T.class_confidence
        )
        if low:
            assert decide(decision_input, T).action is not Action.AUTO_ANSWER, decision_input


def test_invariant_4_clarification_only_on_first_iteration():
    for decision_input in ALL_INPUTS:
        if decide(decision_input, T).action is Action.CLARIFY:
            assert decision_input.clarification_iteration == 0, decision_input
            assert decision_input.category is Category.TECH_ISSUE


def test_invariant_6_order_status_never_auto_answered_without_integration():
    """R10: без данных заказа автоответ формально grounded, но бесполезен."""
    for decision_input in ALL_INPUTS:
        if decision_input.category is Category.ORDER_STATUS:
            assert decide(decision_input, T).action is not Action.AUTO_ANSWER, decision_input


def test_invariant_5_no_gaps_in_decision_table():
    gaps = [i for i in ALL_INPUTS if decide(i, T).rule_id == "R-default"]
    assert gaps == [], f"не покрыто таблицей: {gaps[:5]}"


# --------------------------------------------------------------------------
# Порядок правил (first match wins)
# --------------------------------------------------------------------------


def test_r7b_wins_over_r8_after_clarification():
    """После исчерпания лимита маршрут не зависит от RAG confidence."""
    decision = decide(ti(Category.TECH_ISSUE, rag=0.1, cls=0.1, iteration=1), T)
    assert decision.rule_id == "R7b"
    assert decision.reason is EscalationReason.CLARIFICATION_LIMIT_REACHED


def test_r9_wins_over_confidence_rules():
    assert decide(ti(Category.UNCLASSIFIED, rag=0.99, cls=0.99), T).rule_id == "R9"


def test_r1_wins_over_r9():
    assert decide(ti(Category.UNCLASSIFIED, human=True), T).rule_id == "R1"


# --------------------------------------------------------------------------
# Границы порогов
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("class_confidence", "expected_rule"),
    [(0.8499, "R5"), (0.85, "R4"), (0.8501, "R4")],
)
def test_class_confidence_boundary(class_confidence, expected_rule):
    assert decide(ti(Category.FAQ, rag=0.9, cls=class_confidence), T).rule_id == expected_rule


@pytest.mark.parametrize(
    ("rag_confidence", "expected_rule"),
    [(0.6999, "R6"), (0.7, "R4"), (0.7001, "R4")],
)
def test_rag_confidence_boundary(rag_confidence, expected_rule):
    assert decide(ti(Category.FAQ, rag=rag_confidence, cls=0.95), T).rule_id == expected_rule


def test_missing_measurements_are_treated_as_below_threshold():
    """Отсутствие измерения не должно открывать дорогу к автоответу."""
    assert decide(ti(Category.FAQ, rag=None, cls=0.95), T).rule_id == "R6"
    assert decide(ti(Category.FAQ, rag=0.9, cls=None), T).rule_id == "R5"


# --------------------------------------------------------------------------
# Пороги - параметр, а не константа кода
# --------------------------------------------------------------------------


def test_thresholds_are_configurable_not_hardcoded():
    strict = Thresholds(class_confidence=0.99, rag_confidence=0.95)
    decision_input = ti(Category.FAQ, rag=0.9, cls=0.95)
    assert decide(decision_input, T).rule_id == "R4"
    assert decide(decision_input, strict).rule_id == "R6"


def test_zero_clarifications_disables_a3():
    """max_clarifications = 0 - легальная конфигурация: уточнения выключены."""
    no_clarify = Thresholds(max_clarifications=0)
    decision = decide(ti(Category.TECH_ISSUE, rag=0.95, cls=0.95), no_clarify)
    assert decision.rule_id == "R7b"
    assert decision.action is Action.ESCALATE


# --------------------------------------------------------------------------
# Валидация входа и свойства результата
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"class_confidence": 1.5},
        {"rag_confidence": -0.1},
        {"clarification_iteration": -1},
    ],
)
def test_invalid_input_rejected(kwargs):
    with pytest.raises(ValueError):
        DecisionInput(Category.FAQ, **kwargs)


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        Thresholds(rag_confidence=1.2)


def test_confidences_cannot_be_passed_positionally():
    """Защита от перестановки двух float: всё, кроме категории, - keyword-only."""
    with pytest.raises(TypeError):
        DecisionInput(Category.FAQ, 0.9, 0.95)  # type: ignore[misc]


def test_escalations_carry_reason_and_answers_do_not():
    for decision_input in ALL_INPUTS:
        decision = decide(decision_input, T)
        if decision.is_escalation:
            assert decision.reason is not None, decision_input
        else:
            assert decision.reason is None, decision_input


def test_only_client_requested_escalation_is_prioritized():
    for decision_input in ALL_INPUTS:
        decision = decide(decision_input, T)
        expected = (
            PRIORITY_CLIENT_REQUESTED
            if decision.action is Action.PRIORITY_ESCALATE
            else PRIORITY_STANDARD
        )
        assert decision.priority == expected, decision_input


def test_decide_is_deterministic():
    for decision_input in ALL_INPUTS:
        assert decide(decision_input, T) == decide(decision_input, T)
