import { useEffect, useState } from "react";
import type { Api } from "../api";
import { REASON_LABELS, formatTime, label, waitingFor } from "../labels";
import type { Escalation } from "../types";

const POLL_MS = 30_000;

export function Queue({ api, version, feedLive }: { api: Api; version: number; feedLive: boolean }) {
  const [items, setItems] = useState<Escalation[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  // Перечитываем очередь по push-событию (version) или по таймеру, если push
  // недоступен. Источник истины - REST, сокет только подсказывает «пора».
  useEffect(() => {
    let cancelled = false;
    api
      .queue()
      .then((page) => !cancelled && (setItems(page.items), setError(null)))
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [api, version, tick]);

  useEffect(() => {
    if (feedLive) return;
    const timer = window.setInterval(() => setTick((value) => value + 1), POLL_MS);
    return () => window.clearInterval(timer);
  }, [feedLive]);

  if (error) return <p className="error">Очередь не загрузилась: {error}</p>;
  if (!items) return <p className="muted">Загрузка очереди…</p>;

  return (
    <section>
      <div className="section-head">
        <h2>Очередь эскалаций</h2>
        <span className="muted">{items.length ? `${items.length} в ожидании` : ""}</span>
      </div>

      {items.length === 0 ? (
        <div className="empty card">Очередь пуста - агент справляется сам.</div>
      ) : (
        <ul className="queue">
          {items.map((item) => (
            <li key={item.id}>
              <a className="queue-item card" href={`#/escalations/${item.id}`}>
                <span className={`priority ${item.priority > 0 ? "priority-high" : ""}`}>
                  {item.priority > 0 ? "Приоритет" : "Обычная"}
                </span>
                <span className="reason">{label(REASON_LABELS, item.reason)}</span>
                <span className="meta">
                  {item.rule_id && <span className="rule">{item.rule_id}</span>}
                  {item.draft_text ? (
                    <span className="tag">есть черновик</span>
                  ) : (
                    <span className="tag tag-muted">без черновика</span>
                  )}
                  {item.status === "in_progress" && <span className="tag tag-warn">claim истёк</span>}
                </span>
                <span className="wait" title={formatTime(item.created_at)}>
                  ждёт {waitingFor(item.created_at)}
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
