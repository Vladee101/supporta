import { useEffect, useRef, useState } from "react";

export type FeedStatus = "connecting" | "live" | "offline";

export interface FeedEvent {
  type: string;
  escalation_id?: string;
  priority?: number;
  reason?: string;
}

const RECONNECT_MS = 3000;

/**
 * Подписка на push эскалаций. Событие - подсказка «очередь изменилась», а не
 * данные: получив его, консоль перечитывает очередь по REST. Поэтому потерянное
 * при обрыве событие ничего не ломает - после переподключения очередь всё
 * равно перечитывается.
 */
export function useEscalationFeed(token: string, onEvent: (event: FeedEvent) => void): FeedStatus {
  const [status, setStatus] = useState<FeedStatus>("connecting");
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let timer: number | undefined;
    let stopped = false;

    const connect = () => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      // Токен в query: браузерный WebSocket не умеет ставить заголовок Authorization.
      socket = new WebSocket(
        `${scheme}://${location.host}/api/v1/ws/escalations?token=${encodeURIComponent(token)}`,
      );
      socket.onopen = () => {
        setStatus("live");
        handler.current({ type: "reconnected" });
      };
      socket.onmessage = (message) => {
        try {
          handler.current(JSON.parse(message.data) as FeedEvent);
        } catch {
          // Некорректное уведомление игнорируем: источник истины - REST.
        }
      };
      socket.onclose = () => {
        setStatus("offline");
        if (!stopped) timer = window.setTimeout(connect, RECONNECT_MS);
      };
    };

    connect();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket?.close();
    };
  }, [token]);

  return status;
}
