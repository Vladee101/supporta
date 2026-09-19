"""Retention сырых тикетов (NFR4): персональные данные живут не дольше срока хранения.

Строки тикетов не удаляются, а вычищаются. Удаление строки тикета каскадом
унесло бы `audit_log` (FK `ON DELETE CASCADE`), а требование - «audit_log и
агрегаты сохраняются». Поэтому по истечении срока заменяется всё, что может
содержать персональные данные:

* `messages.content` - оригинал обращения (PII-редактор не ловит имена и
  адреса, так что и `content_redacted` не считается обезличенным);
* `escalations.draft_text` и `operator_actions.draft_text / final_text` -
  тексты для клиента могут повторять его имя и детали;
* `tickets.client_id` - идентификатор клиента в канале.

Остаются категория, статус, правила, оба confidence, время, трейс и сходство
черновика с ответом оператора - всё, на чём держатся NFR3 и метрики качества.

Открытые тикеты не вычищаются, даже если старше срока: оператор, который
работает с тикетом, должен видеть обращение. Такие тикеты возвращаются в
результате как нарушение NFR4 и попадают в лог шедулера - это сигнал разобрать
зависшую эскалацию, а не повод молча уничтожить рабочие данные.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Escalation, Message, OperatorAction, Ticket
from app.domain.enums import TicketStatus

TOMBSTONE = "[удалено: истёк срок хранения]"

CLOSED_STATUSES = (TicketStatus.RESOLVED_AUTO.value, TicketStatus.RESOLVED_BY_OPERATOR.value)
DEFAULT_BATCH = 500


@dataclass(frozen=True, slots=True)
class RetentionResult:
    scrubbed: int
    #: Открытые тикеты старше срока хранения - нарушение NFR4, требует разбора.
    overdue_open: int


def scrub_expired(
    session: Session,
    *,
    now: datetime,
    retention: timedelta,
    batch_size: int = DEFAULT_BATCH,
) -> RetentionResult:
    cutoff = now - retention

    ticket_ids: list[uuid.UUID] = list(
        session.scalars(
            select(Ticket.id)
            .where(
                Ticket.created_at < cutoff,
                Ticket.scrubbed_at.is_(None),
                Ticket.status.in_(CLOSED_STATUSES),
            )
            .order_by(Ticket.created_at)
            .limit(batch_size)
            # Несколько экземпляров шедулера не вычищают одно и то же.
            .with_for_update(skip_locked=True)
        )
    )

    if ticket_ids:
        _scrub(session, ticket_ids, now)

    overdue_open = session.scalar(
        select(func.count())
        .select_from(Ticket)
        .where(
            Ticket.created_at < cutoff,
            Ticket.scrubbed_at.is_(None),
            Ticket.status.not_in(CLOSED_STATUSES),
        )
    )
    session.commit()
    if ticket_ids:
        # Массовые UPDATE не трогают объекты, уже загруженные в сессию: без
        # сброса вызывающий код видел бы в памяти персональные данные, которых
        # в базе уже нет.
        session.expire_all()
    return RetentionResult(scrubbed=len(ticket_ids), overdue_open=int(overdue_open or 0))


def _scrub(session: Session, ticket_ids: list[uuid.UUID], now: datetime) -> None:
    session.execute(
        update(Message)
        .where(Message.ticket_id.in_(ticket_ids))
        .values(content=TOMBSTONE, content_redacted=None)
        .execution_options(synchronize_session=False)
    )

    escalation_ids = select(Escalation.id).where(Escalation.ticket_id.in_(ticket_ids))
    session.execute(
        update(OperatorAction)
        .where(OperatorAction.escalation_id.in_(escalation_ids))
        .values(
            # NULL остаётся NULL: «черновика не было» (ADR-008) - это метрика, а не PII.
            draft_text=case((OperatorAction.draft_text.is_(None), None), else_=TOMBSTONE),
            final_text=TOMBSTONE,
        )
        .execution_options(synchronize_session=False)
    )
    session.execute(
        update(Escalation)
        .where(Escalation.ticket_id.in_(ticket_ids))
        .values(draft_text=None)
        .execution_options(synchronize_session=False)
    )
    session.execute(
        update(Ticket)
        .where(Ticket.id.in_(ticket_ids))
        .values(client_id=None, scrubbed_at=now)
        .execution_options(synchronize_session=False)
    )
    session.add_all(
        AuditLog(
            ticket_id=ticket_id,
            actor="scheduler",
            action="retention_scrub",
            payload={"scrubbed_at": now.isoformat()},
        )
        for ticket_id in ticket_ids
    )
