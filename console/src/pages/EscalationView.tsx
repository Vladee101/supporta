import { useCallback, useEffect, useState } from "react";
import { ApiError, type Api } from "../api";
import { CATEGORY_LABELS, REASON_LABELS, SENDER_LABELS, formatTime, label, waitingFor } from "../labels";
import type { AuditTrail, EscalationContext, ResolveAction } from "../types";

type Tab = "context" | "audit";

export function EscalationView({ api, id, operatorId }: { api: Api; id: string; operatorId: string }) {
  const [data, setData] = useState<EscalationContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("context");

  const reload = useCallback(() => {
    api
      .context(id)
      .then((next) => (setData(next), setError(null)))
      .catch((err: Error) => setError(err.message));
  }, [api, id]);

  useEffect(reload, [reload]);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Загрузка…</p>;

  const { escalation, ticket } = data;

  return (
    <section>
      <a className="back" href="#/queue">
        ← Очередь
      </a>
      <div className="section-head">
        <h2>{label(REASON_LABELS, escalation.reason)}</h2>
        <span className={`priority ${escalation.priority > 0 ? "priority-high" : ""}`}>
          {escalation.priority > 0 ? "Приоритет" : "Обычная"}
        </span>
      </div>
      <p className="muted facts">
        <span>Правило {escalation.rule_id ?? "—"}</span>
        <span>Категория: {label(CATEGORY_LABELS, ticket.category)}</span>
        <span>Канал: {ticket.channel}</span>
        <span>Уточнений: {ticket.clarification_count}</span>
        <span>Ждёт {waitingFor(escalation.created_at)}</span>
      </p>

      <div className="tabs">
        <button className={tab === "context" ? "active" : ""} onClick={() => setTab("context")}>
          Контекст
        </button>
        <button className={tab === "audit" ? "active" : ""} onClick={() => setTab("audit")}>
          Audit trail
        </button>
      </div>

      {tab === "context" ? (
        <div className="columns">
          <div className="column-main">
            <Conversation data={data} />
            <ActionPanel api={api} data={data} operatorId={operatorId} onChange={reload} />
          </div>
          <aside className="column-side">
            <AgentReasoning data={data} />
          </aside>
        </div>
      ) : (
        <Audit api={api} ticketId={ticket.id} />
      )}
    </section>
  );
}

function Conversation({ data }: { data: EscalationContext }) {
  return (
    <div className="card">
      <h3>Переписка</h3>
      <ol className="messages">
        {data.messages.map((message, index) => (
          <li key={index} className={`message message-${message.sender}`}>
            <div className="message-head">
              <strong>{label(SENDER_LABELS, message.sender)}</strong>
              <span className="muted">{formatTime(message.created_at)}</span>
              {message.iteration > 0 && <span className="tag">итерация {message.iteration}</span>}
            </div>
            <p>{message.content}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}

function AgentReasoning({ data }: { data: EscalationContext }) {
  return (
    <>
      <div className="card">
        <h3>Классификация</h3>
        {data.classifications.length === 0 && (
          <p className="muted">Не выполнялась: клиент сразу запросил оператора (R1).</p>
        )}
        {data.classifications.map((row) => (
          <div key={row.iteration} className="classification">
            <div className="classification-head">
              <strong>{label(CATEGORY_LABELS, row.category)}</strong>
              <span className="muted">итерация {row.iteration}</span>
            </div>
            <Confidence value={row.confidence} />
            <p className="muted small">
              {row.confidence_source ?? "—"} · {row.model_id ?? "—"}
            </p>
            {row.reasoning && <p className="small">{row.reasoning}</p>}
          </div>
        ))}
      </div>

      <div className="card">
        <h3>Найденные документы</h3>
        {data.documents.length === 0 && <p className="muted">RAG не нашёл документов.</p>}
        {data.documents.map((doc) => (
          <details key={`${doc.iteration}-${doc.rank}`} className="document">
            <summary>
              {doc.title}
              <span className="muted small">
                {" "}· v{doc.version} · релевантность {doc.relevance_score.toFixed(2)}
                {doc.iteration > 0 && <> · итерация {doc.iteration}</>}
              </span>
            </summary>
            {/* Снапшот того текста, что видел агент, - не текущая версия документа (ADR-011). */}
            <p className="small">{doc.snapshot}</p>
          </details>
        ))}
      </div>
    </>
  );
}

function Confidence({ value }: { value: number | null }) {
  if (value === null) return <p className="muted small">confidence не измерен</p>;
  return (
    <div className="confidence" title={value.toFixed(3)}>
      <div className="confidence-bar" style={{ width: `${Math.round(value * 100)}%` }} />
      <span>{value.toFixed(2)}</span>
    </div>
  );
}

function ActionPanel({
  api,
  data,
  operatorId,
  onChange,
}: {
  api: Api;
  data: EscalationContext;
  operatorId: string;
  onChange: () => void;
}) {
  const { escalation } = data;
  const draft = escalation.draft_text;
  const [text, setText] = useState(draft ?? "");
  const [fromScratch, setFromScratch] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<string | null>(null);

  const lockExpired = escalation.locked_until !== null && new Date(escalation.locked_until) < new Date();
  const mine = escalation.locked_by === operatorId && !lockExpired;
  const takenByOther = escalation.locked_by !== null && !mine && !lockExpired;

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      onChange();
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (escalation.status === "resolved") {
    return <div className="card done">Эскалация закрыта {escalation.resolved_at && formatTime(escalation.resolved_at)}.</div>;
  }
  if (sent) return <div className="card done">Ответ отправлен клиенту.</div>;

  if (!mine) {
    return (
      <div className="card actions">
        {takenByOther ? (
          <p>В работе у другого оператора до {formatTime(escalation.locked_until!)}.</p>
        ) : (
          <button className="primary" disabled={busy} onClick={() => run(() => api.claim(escalation.id))}>
            Взять в работу
          </button>
        )}
        {error && <p className="error">{error}</p>}
      </div>
    );
  }

  // Действие выводится из того, что сделал оператор: без правок черновика -
  // confirm, с правками - edit, свой текст вместо черновика - reject (NFR7).
  const unchangedDraft = draft !== null && !fromScratch && text.trim() === draft.trim();
  const action: ResolveAction = draft === null || fromScratch ? "reject" : unchangedDraft ? "confirm" : "edit";
  const buttonLabel = {
    confirm: "Отправить черновик",
    edit: "Отправить с правками",
    reject: "Отправить свой ответ",
  }[action];

  return (
    <div className="card actions">
      <div className="actions-head">
        <h3>Ответ клиенту</h3>
        <span className="muted small">claim до {formatTime(escalation.locked_until!)}</span>
      </div>
      {draft === null && <p className="muted small">{noDraftReason(escalation.reason)}</p>}
      <textarea
        rows={6}
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Текст ответа клиенту"
      />
      <div className="buttons">
        <button
          className="primary"
          disabled={busy || !text.trim()}
          onClick={() =>
            run(async () => {
              await api.resolve(escalation.id, action, action === "confirm" ? undefined : text);
              setSent(text);
            })
          }
        >
          {buttonLabel}
        </button>
        {draft !== null && !fromScratch && (
          <button
            disabled={busy}
            onClick={() => {
              setFromScratch(true);
              setText("");
            }}
          >
            Написать с нуля
          </button>
        )}
        {draft !== null && (fromScratch || !unchangedDraft) && (
          <button
            className="link"
            disabled={busy}
            onClick={() => {
              setFromScratch(false);
              setText(draft);
            }}
          >
            Вернуть черновик
          </button>
        )}
        <button className="link" disabled={busy} onClick={() => run(() => api.release(escalation.id))}>
          Вернуть в очередь
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

/** Почему черновика нет - у причин разная природа, и оператору важно её видеть. */
function noDraftReason(reason: string): string {
  if (reason === "high_risk_category") {
    return "Для жалоб и возвратов черновик не готовится (ADR-008): ответ пишется с нуля.";
  }
  if (reason === "client_requested") {
    return "Клиент попросил человека после ответа агента - агент черновик не готовил, его ответ виден в переписке.";
  }
  return "Черновика нет: ответ пишется с нуля.";
}

function Audit({ api, ticketId }: { api: Api; ticketId: string }) {
  const [trail, setTrail] = useState<AuditTrail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .audit(ticketId)
      .then(setTrail)
      .catch((err: Error) => setError(err.message));
  }, [api, ticketId]);

  if (error) return <p className="error">{error}</p>;
  if (!trail) return <p className="muted">Загрузка…</p>;

  return (
    <div className="card">
      {trail.clarification_limit_reached && (
        <p className="tag tag-warn">Сработал лимит уточнений (NFR9)</p>
      )}
      <ol className="timeline">
        {trail.events.map((event, index) => (
          <li key={index}>
            <div className="timeline-head">
              <span className="muted">{formatTime(event.at)}</span>
              <strong>{event.actor}</strong>
              <span>{event.action}</span>
              {event.rule_id && <span className="rule">{event.rule_id}</span>}
            </div>
            {(event.class_confidence !== null || event.rag_confidence !== null) && (
              <p className="small muted">
                класс. confidence {event.class_confidence?.toFixed(2) ?? "—"} · RAG confidence{" "}
                {event.rag_confidence?.toFixed(2) ?? "—"}
              </p>
            )}
            {event.reasoning && <p className="small">{event.reasoning}</p>}
            {event.payload && (
              <details>
                <summary className="small muted">подробности{event.trace_id ? ` · trace ${event.trace_id.slice(0, 8)}` : ""}</summary>
                <pre>{JSON.stringify(event.payload, null, 2)}</pre>
              </details>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
