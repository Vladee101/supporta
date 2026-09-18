"""Decision Engine - детерминированная маршрутизация тикета.

Реализация decision table (R1-R9, R7c, R-default) из design document.

Три свойства, ради которых движок написан кодом, а не промптом (ADR-001):

* **чистота** - никакого I/O, никакого обращения к LLM: вход `DecisionInput`,
  выход `Decision`. Одинаковый вход всегда даёт одинаковый выход, поэтому
  решение воспроизводимо при разборе инцидента;
* **порядок как часть спецификации** - правила проверяются сверху вниз,
  срабатывает первое совпавшее. Без явного порядка R7b и R8 перекрываются;
* **отсутствие поведения по умолчанию** - непокрытая комбинация не превращается
  в тихий автоответ, а даёт R-default: эскалацию плюс алерт `decision_table_gap`.

Пороги приходят снаружи (`Thresholds`), в предикатах правил числовых констант нет:
0.85 и 0.7 калибруются на golden set и меняются вместе с моделью, а не с кодом.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.domain.enums import (
    AUTO_ANSWERABLE_CATEGORIES,
    HIGH_RISK_CATEGORIES,
    Action,
    Category,
    EscalationReason,
)

#: Приоритет эскалации по явному запросу клиента (A4) - выше стандартной очереди (UC9).
PRIORITY_CLIENT_REQUESTED = 10
PRIORITY_STANDARD = 0


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Калибруемые пороги. Значения по умолчанию - стартовые, см. «Confidence и пороги»."""

    class_confidence: float = 0.85
    rag_confidence: float = 0.7
    max_clarifications: int = 1

    def __post_init__(self) -> None:
        for name in ("class_confidence", "rag_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} должен быть в [0, 1], получено {value!r}")
        if self.max_clarifications < 0:
            raise ValueError("max_clarifications не может быть отрицательным")


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """Полный вход движка. Всё, что влияет на маршрут, перечислено здесь явно.

    `class_confidence` / `rag_confidence` могут быть None: классификация не
    состоялась или RAG-поиск не выполнялся. None трактуется как «ниже порога» -
    отсутствие измерения не должно открывать дорогу к автоответу.

    Все поля, кроме категории, keyword-only: два confidence - это два float в
    одном диапазоне, при позиционной передаче их можно молча поменять местами,
    и ни одна проверка диапазона такую ошибку не поймает.
    """

    category: Category
    rag_confidence: float | None = field(default=None, kw_only=True)
    class_confidence: float | None = field(default=None, kw_only=True)
    clarification_iteration: int = field(default=0, kw_only=True)
    human_requested: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        for name in ("class_confidence", "rag_confidence"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} должен быть в [0, 1] или None, получено {value!r}")
        if self.clarification_iteration < 0:
            raise ValueError("clarification_iteration не может быть отрицательным")


@dataclass(frozen=True, slots=True)
class Decision:
    """Результат маршрутизации.

    `rule_id` пишется в `audit_log.rule_id` - по трейсу видно не только что решил
    агент, но и какая строка таблицы сработала (NFR3).
    """

    action: Action
    rule_id: str
    reason: EscalationReason | None = None
    priority: int = PRIORITY_STANDARD
    alert: str | None = None

    @property
    def is_escalation(self) -> bool:
        return self.action in (Action.ESCALATE, Action.PRIORITY_ESCALATE)


Predicate = Callable[[DecisionInput, Thresholds], bool]


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    predicate: Predicate
    action: Action
    reason: EscalationReason | None = None
    priority: int = PRIORITY_STANDARD
    description: str = field(default="", compare=False)


def _class_confident(i: DecisionInput, t: Thresholds) -> bool:
    return i.class_confidence is not None and i.class_confidence >= t.class_confidence


def _rag_confident(i: DecisionInput, t: Thresholds) -> bool:
    return i.rag_confidence is not None and i.rag_confidence >= t.rag_confidence


def _clarification_exhausted(i: DecisionInput, t: Thresholds) -> bool:
    return i.clarification_iteration >= t.max_clarifications


def _is_tech(i: DecisionInput) -> bool:
    return i.category is Category.TECH_ISSUE


def _is_auto_answerable(i: DecisionInput) -> bool:
    return i.category in AUTO_ANSWERABLE_CATEGORIES


#: Порядок строк - часть спецификации: срабатывает первое совпавшее правило.
RULES: tuple[Rule, ...] = (
    Rule(
        "R1",
        lambda i, t: i.human_requested,
        Action.PRIORITY_ESCALATE,
        EscalationReason.CLIENT_REQUESTED,
        PRIORITY_CLIENT_REQUESTED,
        "Явный запрос человека побеждает всё остальное (UC9)",
    ),
    Rule(
        "R9",
        lambda i, t: i.category is Category.UNCLASSIFIED,
        Action.ESCALATE,
        EscalationReason.CLASSIFICATION_FAILED,
        description="Классификация не удалась",
    ),
    Rule(
        "R2/R3",
        lambda i, t: i.category in HIGH_RISK_CATEGORIES,
        Action.ESCALATE,
        EscalationReason.HIGH_RISK_CATEGORY,
        description="Жалоба / возврат денег - никогда не автоответ, confidence не влияет",
    ),
    # Временное правило: без интеграции с системой заказов агент не знает статус
    # конкретного заказа и отвечает общим документом о статусах - формально
    # grounded, по сути бесполезно. Снимается вместе с появлением интеграции
    # (roadmap v1.1); тогда статус заказа снова пойдёт по R5/R6/R4.
    Rule(
        "R10",
        lambda i, t: i.category is Category.ORDER_STATUS,
        Action.ESCALATE,
        EscalationReason.ORDER_DATA_UNAVAILABLE,
        description="Статус заказа - на оператора, пока нет интеграции с системой заказов",
    ),
    Rule(
        "R7b",
        lambda i, t: _is_tech(i) and _clarification_exhausted(i, t),
        Action.ESCALATE,
        EscalationReason.CLARIFICATION_LIMIT_REACHED,
        description="Лимит уточнений исчерпан (NFR9) - принудительная эскалация",
    ),
    Rule(
        "R7c",
        lambda i, t: _is_tech(i) and _rag_confident(i, t) and not _class_confident(i, t),
        Action.ESCALATE,
        EscalationReason.LOW_CLASS_CONFIDENCE,
        description="Техпроблема с низким class confidence - без попытки уточнения (UC2)",
    ),
    Rule(
        "R7a",
        lambda i, t: _is_tech(i) and _rag_confident(i, t) and _class_confident(i, t),
        Action.CLARIFY,
        description="Техпроблема, 0-я итерация - запрос уточнения",
    ),
    Rule(
        "R8",
        lambda i, t: _is_tech(i) and not _rag_confident(i, t),
        Action.ESCALATE,
        EscalationReason.LOW_RAG_CONFIDENCE,
        description="Техпроблема без опоры в базе знаний",
    ),
    Rule(
        "R5",
        lambda i, t: _is_auto_answerable(i) and _rag_confident(i, t) and not _class_confident(i, t),
        Action.ESCALATE,
        EscalationReason.LOW_CLASS_CONFIDENCE,
        description="FAQ / статус заказа с низким class confidence",
    ),
    Rule(
        "R6",
        lambda i, t: _is_auto_answerable(i) and not _rag_confident(i, t),
        Action.ESCALATE,
        EscalationReason.LOW_RAG_CONFIDENCE,
        description="FAQ / статус заказа без опоры в базе знаний",
    ),
    Rule(
        "R4",
        lambda i, t: _is_auto_answerable(i) and _rag_confident(i, t) and _class_confident(i, t),
        Action.AUTO_ANSWER,
        description="Единственный путь к автономному ответу клиенту",
    ),
)

#: Fail-safe: у движка нет тихого поведения по умолчанию.
DEFAULT_RULE = Rule(
    "R-default",
    lambda i, t: True,
    Action.ESCALATE,
    EscalationReason.DECISION_TABLE_GAP,
    description="Комбинация не покрыта таблицей - эскалация и алерт как дефект таблицы",
)

DEFAULT_THRESHOLDS = Thresholds()


def decide(decision_input: DecisionInput, thresholds: Thresholds | None = None) -> Decision:
    """Вернуть решение по тикету. Чистая функция: без I/O и без побочных эффектов."""
    t = thresholds or DEFAULT_THRESHOLDS

    for rule in RULES:
        if rule.predicate(decision_input, t):
            return Decision(
                action=rule.action,
                rule_id=rule.rule_id,
                reason=rule.reason,
                priority=rule.priority,
            )

    return Decision(
        action=DEFAULT_RULE.action,
        rule_id=DEFAULT_RULE.rule_id,
        reason=DEFAULT_RULE.reason,
        priority=PRIORITY_STANDARD,
        alert="decision_table_gap",
    )
