"""Доменные перечисления.

Значения строковые и совпадают с тем, что пишется в Postgres и в audit_log,
чтобы трейс читался без обратного маппинга (NFR3).
"""

from __future__ import annotations

from enum import StrEnum


class Category(StrEnum):
    """Категории тикета из decision table."""

    FAQ = "faq"
    ORDER_STATUS = "order_status"
    COMPLAINT = "complaint"
    REFUND = "refund"
    TECH_ISSUE = "tech_issue"
    UNCLASSIFIED = "unclassified"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(StrEnum):
    """Действия A1-A4 из decision table."""

    AUTO_ANSWER = "A1"
    ESCALATE = "A2"
    CLARIFY = "A3"
    PRIORITY_ESCALATE = "A4"


class EscalationReason(StrEnum):
    """Причина эскалации. Пишется в escalations.reason и в audit_log."""

    CLIENT_REQUESTED = "client_requested"
    HIGH_RISK_CATEGORY = "high_risk_category"
    CLASSIFICATION_FAILED = "classification_failed"
    LOW_CLASS_CONFIDENCE = "low_class_confidence"
    LOW_RAG_CONFIDENCE = "low_rag_confidence"
    CLARIFICATION_LIMIT_REACHED = "clarification_limit_reached"
    CLARIFICATION_TIMEOUT = "clarification_timeout"
    AGENT_TIMEOUT = "agent_timeout"
    LLM_UNAVAILABLE = "llm_unavailable"
    #: R10: нет интеграции с системой заказов - агент не знает статус заказа.
    ORDER_DATA_UNAVAILABLE = "order_data_unavailable"
    DECISION_TABLE_GAP = "decision_table_gap"


class TicketStatus(StrEnum):
    """Состояния из диаграммы состояний тикета."""

    NEW = "new"
    CLASSIFIED = "classified"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    ESCALATED_STANDARD = "escalated_standard"
    ESCALATED_PRIORITY = "escalated_priority"
    PENDING_OPERATOR = "pending_operator"
    IN_PROGRESS = "in_progress"
    RESOLVED_AUTO = "resolved_auto"
    RESOLVED_BY_OPERATOR = "resolved_by_operator"


class EscalationStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class OperatorActionType(StrEnum):
    CONFIRM = "confirm"
    EDIT = "edit"
    REJECT = "reject"


class MessageSender(StrEnum):
    CLIENT = "client"
    AGENT = "agent"
    OPERATOR = "operator"


class ConfidenceSource(StrEnum):
    """Как получен class_confidence - см. «Confidence и пороги».

    Хранится рядом со значением: при смене провайдера шкала меняется
    и пороги перестают означать прежнее.
    """

    LOGPROBS = "logprobs"
    K_SAMPLING = "k_sampling"
    #: Словарная базовая линия - своя шкала, пороги LLM к ней неприменимы.
    BASELINE = "baseline"
    #: Уверенность LLM, пониженная при расхождении с базовой линией (ADR-012).
    CROSS_CHECK = "cross_check"


class OperatorRole(StrEnum):
    OPERATOR = "operator"
    ADMIN = "admin"


#: Категории, которые никогда не отвечаются автономно (инвариант 2 decision table).
HIGH_RISK_CATEGORIES: frozenset[Category] = frozenset({Category.COMPLAINT, Category.REFUND})

#: Категории, для которых автоответ в принципе возможен (R4).
AUTO_ANSWERABLE_CATEGORIES: frozenset[Category] = frozenset({Category.FAQ, Category.ORDER_STATUS})

#: Категория, уходящая в цикл уточнения (R7a).
CLARIFIABLE_CATEGORIES: frozenset[Category] = frozenset({Category.TECH_ISSUE})

RISK_BY_CATEGORY: dict[Category, RiskLevel] = {
    Category.FAQ: RiskLevel.LOW,
    Category.ORDER_STATUS: RiskLevel.LOW,
    Category.TECH_ISSUE: RiskLevel.MEDIUM,
    Category.COMPLAINT: RiskLevel.HIGH,
    Category.REFUND: RiskLevel.HIGH,
    Category.UNCLASSIFIED: RiskLevel.HIGH,
}
