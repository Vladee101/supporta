"""Граф обработки тикета (LangGraph, ADR-002).

    redact → [human_requested?] → classify → retrieve → decide → act

Две вещи, ради которых граф выглядит именно так:

* **R1 проверяется первым.** Если клиент явно попросил человека, классификация
  и RAG не выполняются вообще: агент прекращает автоматические попытки ответа
  (UC9), а заодно экономит два LLM-вызова из бюджета NFR1/NFR5.
* **Маршрут выбирает Decision Engine, а не узел графа.** `decide_node` -
  единственное место, где принимается решение, и он вызывает чистую функцию
  `decide()`. Узлы до него только собирают измерения, узел после - исполняет
  уже принятое решение.

Граф не пишет в базу: он возвращает результат, а персистентность и транзакция
живут в `app.agent.service`. Так пайплайн тестируется без Postgres, а запись
тикета, эскалации и outbox остаётся одной транзакцией (ADR-004, ADR-007).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.db.base import release_connection
from app.domain.decision import Decision, DecisionInput, Thresholds, decide
from app.domain.enums import HIGH_RISK_CATEGORIES, Action, Category
from app.services import llm_usage
from app.services.classifier import ClassificationResult, Classifier
from app.services.generation import Draft, ResponseGenerator
from app.services.llm import LLMUnavailableError
from app.services.pii import RedactionResult, redact
from app.services.retrieval import RetrievalResult, Retriever

CLARIFICATION_TEMPLATE = (
    "Чтобы разобраться точнее, уточните, пожалуйста, детали: "
    "что именно происходит, на каком шаге и какое устройство или браузер вы используете."
)


class AgentState(TypedDict, total=False):
    # Вход
    ticket_text: str
    iteration: int
    human_requested: bool
    session: Session
    # Промежуточные измерения
    redaction: RedactionResult
    classification: ClassificationResult | None
    retrieval: RetrievalResult | None
    # Выход
    decision: Decision
    draft: Draft | None
    reply_text: str | None


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """Что агент решил и на основании чего. Всё нужное для audit_log (NFR3)."""

    decision: Decision
    redaction: RedactionResult
    classification: ClassificationResult | None
    retrieval: RetrievalResult | None
    draft: Draft | None
    reply_text: str | None
    #: Расход LLM на этот проход графа (NFR5): вызовы, токены, стоимость от провайдера.
    llm_usage: dict[str, Any] | None = None

    @property
    def category(self) -> Category:
        return self.classification.category if self.classification else Category.UNCLASSIFIED


class TicketGraph:
    """Собранный граф. Зависимости внедряются - провайдеры подменяемы (ADR-009)."""

    def __init__(
        self,
        classifier: Classifier,
        retriever: Retriever,
        generator: ResponseGenerator,
        thresholds: Thresholds | None = None,
    ) -> None:
        self._classifier = classifier
        self._retriever = retriever
        self._generator = generator
        self._thresholds = thresholds or Thresholds()
        self._graph = self._build()

    # --- узлы ---------------------------------------------------------------

    def _redact_node(self, state: AgentState) -> dict[str, Any]:
        """NFR4: дальше по графу идёт только замаскированный текст."""
        return {"redaction": redact(state["ticket_text"])}

    def _classify_node(self, state: AgentState) -> dict[str, Any]:
        return {"classification": self._classifier.classify(state["redaction"].text)}

    def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        # Ищем контекст и для high-risk категорий: маршрут он не изменит (R2/R3),
        # но попадёт в пакет эскалации и сэкономит оператору поиск (UC5).
        session = state["session"]
        retrieval = self._retriever.retrieve(session, state["redaction"].text)
        # Впереди генерация через LLM - соединение возвращается в пул до записи
        # результата. Иначе под нагрузкой пул нужен размером с число тикетов (NFR8).
        if session is not None:
            release_connection(session)
        return {"retrieval": retrieval}

    def _decide_node(self, state: AgentState) -> dict[str, Any]:
        classification = state.get("classification")
        retrieval = state.get("retrieval")
        decision_input = DecisionInput(
            classification.category if classification else Category.UNCLASSIFIED,
            rag_confidence=retrieval.rag_confidence if retrieval else None,
            class_confidence=classification.confidence if classification else None,
            clarification_iteration=state.get("iteration", 0),
            human_requested=state.get("human_requested", False),
        )
        return {"decision": decide(decision_input, self._thresholds)}

    def _act_node(self, state: AgentState) -> dict[str, Any]:
        decision: Decision = state["decision"]
        retrieval = state.get("retrieval")
        chunks = retrieval.chunks if retrieval else ()

        if decision.action is Action.AUTO_ANSWER:
            draft = self._generator.generate(state["redaction"].text, chunks)
            return {"draft": draft, "reply_text": draft.text}

        if decision.action is Action.CLARIFY:
            return {"draft": None, "reply_text": CLARIFICATION_TEMPLATE}

        # Эскалация: черновик готовится не для всех категорий (ADR-008) -
        # на жалобах и возвратах шаблонный текст рискован, оператор пишет сам.
        classification = state.get("classification")
        category = classification.category if classification else Category.UNCLASSIFIED
        if category in HIGH_RISK_CATEGORIES:
            return {"draft": None, "reply_text": None}

        try:
            draft = self._generator.generate(state["redaction"].text, chunks)
        except LLMUnavailableError:
            # Черновик - помощь оператору, а не условие эскалации. Сбой LLM на
            # этом шаге не должен подменять настоящую причину эскалации
            # (скажем, R7b) на «LLM недоступен»: эскалация уходит без черновика.
            # Для автоответа (A1) такой поблажки нет - там без текста ответить
            # нечем, и ошибка поднимается выше, к эскалации по NFR6.
            draft = None
        return {"draft": draft, "reply_text": None}

    # --- сборка -------------------------------------------------------------

    @staticmethod
    def _after_redact(state: AgentState) -> str:
        return "decide" if state.get("human_requested") else "classify"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("redact", self._redact_node)
        graph.add_node("classify", self._classify_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("decide", self._decide_node)
        graph.add_node("act", self._act_node)

        graph.set_entry_point("redact")
        graph.add_conditional_edges(
            "redact", self._after_redact, {"classify": "classify", "decide": "decide"}
        )
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "decide")
        graph.add_edge("decide", "act")
        graph.add_edge("act", END)
        return graph.compile()

    # --- запуск -------------------------------------------------------------

    def run(
        self,
        session: Session,
        ticket_text: str,
        *,
        iteration: int = 0,
        human_requested: bool = False,
    ) -> AgentOutcome:
        with llm_usage.metered() as meter:
            final: AgentState = self._graph.invoke(
                {
                    "ticket_text": ticket_text,
                    "iteration": iteration,
                    "human_requested": human_requested,
                    "session": session,
                }
            )
        return AgentOutcome(
            decision=final["decision"],
            redaction=final["redaction"],
            classification=final.get("classification"),
            retrieval=final.get("retrieval"),
            draft=final.get("draft"),
            reply_text=final.get("reply_text"),
            llm_usage=meter.snapshot(),
        )
