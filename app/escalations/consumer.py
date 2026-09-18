"""Обработка события `escalation.created` на стороне consumer'а.

Порядок - часть корректности (ADR-004):

    1. INSERT в consumed_events (ON CONFLICT DO NOTHING)
    2. перевод тикета в очередь оператора + запись в audit_log
    3. COMMIT
    4. ack брокеру - только после commit'а

Упадём между 3 и 4 - брокер доставит повторно, шаг 1 не вставит строку, и
повтор станет no-op. Упадём до 3 - транзакция откатится целиком, повторная
доставка обработает событие с нуля. Ни в одном случае эскалация не
теряется и не обрабатывается дважды.

Функция не знает про RabbitMQ: транспорт - забота воркера. Поэтому её можно
проверить на реальном Postgres без брокера.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import AuditLog, ConsumedEvent, Escalation, Ticket
from app.domain.enums import TicketStatus

CONSUMER_NAME = "escalation-consumer"

#: Из каких состояний тикет переводится в очередь оператора. Если тикет уже
#: дальше (оператор успел взять его по REST раньше, чем пришло событие),
#: состояние не откатывается назад.
_QUEUEABLE = {TicketStatus.ESCALATED_STANDARD.value, TicketStatus.ESCALATED_PRIORITY.value}


class ConsumeResult(StrEnum):
    PROCESSED = "processed"
    DUPLICATE = "duplicate"


class PoisonMessageError(ValueError):
    """Сообщение нельзя обработать никогда - оно уходит в DLX, а не в requeue."""


def handle_escalation_created(
    session: Session, *, idempotency_key: str | None, payload: dict
) -> ConsumeResult:
    if not idempotency_key:
        raise PoisonMessageError("сообщение без message_id: дедупликация невозможна")

    try:
        escalation_id = uuid.UUID(str(payload["escalation_id"]))
    except (KeyError, ValueError) as exc:
        raise PoisonMessageError(f"некорректный payload: {payload!r}") from exc

    # RETURNING, а не rowcount: через ORM-сессию этот INSERT возвращает
    # rowcount = -1 и при вставке, и при конфликте, а -1 истинно - проверка
    # «вставилось ли» по rowcount пропускала бы каждый дубликат.
    inserted = session.execute(
        insert(ConsumedEvent)
        .values(idempotency_key=idempotency_key, consumer=CONSUMER_NAME)
        .on_conflict_do_nothing(index_elements=[ConsumedEvent.idempotency_key])
        .returning(ConsumedEvent.idempotency_key)
    ).first()
    if inserted is None:
        # ON CONFLICT DO NOTHING ничего не записал - откатывать нечего.
        return ConsumeResult.DUPLICATE

    escalation = session.get(Escalation, escalation_id)
    if escalation is None:
        session.rollback()
        raise PoisonMessageError(f"эскалация {escalation_id} не найдена")

    ticket = session.get(Ticket, escalation.ticket_id)
    previous = ticket.status
    if ticket.status in _QUEUEABLE:
        ticket.status = TicketStatus.PENDING_OPERATOR.value

    session.add(
        AuditLog(
            ticket_id=ticket.id,
            actor=CONSUMER_NAME,
            action="queued_for_operator",
            rule_id=escalation.rule_id,
            payload={
                "escalation_id": str(escalation.id),
                "idempotency_key": idempotency_key,
                "status_before": previous,
                "status_after": ticket.status,
                "priority": escalation.priority,
            },
        )
    )
    session.commit()
    return ConsumeResult.PROCESSED
