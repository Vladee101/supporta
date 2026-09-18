// Человекочитаемые подписи для кодов из API. Коды - стабильный контракт,
// подписи - для оператора.

export const REASON_LABELS: Record<string, string> = {
  client_requested: "Клиент попросил оператора",
  high_risk_category: "Жалоба или возврат",
  classification_failed: "Не удалось классифицировать",
  low_class_confidence: "Неуверенная классификация",
  low_rag_confidence: "Нет ответа в базе знаний",
  clarification_limit_reached: "Лимит уточнений исчерпан",
  clarification_timeout: "Клиент не ответил на уточнение",
  agent_timeout: "Таймаут агента",
  llm_unavailable: "LLM недоступен",
  decision_table_gap: "Дефект таблицы решений",
};

export const CATEGORY_LABELS: Record<string, string> = {
  faq: "Общий вопрос",
  order_status: "Статус заказа",
  complaint: "Жалоба",
  refund: "Возврат",
  tech_issue: "Техпроблема",
  unclassified: "Не классифицирован",
};

export const SENDER_LABELS: Record<string, string> = {
  client: "Клиент",
  agent: "Агент",
  operator: "Оператор",
};

export const label = (map: Record<string, string>, key: string | null | undefined) =>
  key ? (map[key] ?? key) : "—";

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function waitingFor(iso: string, now = Date.now()): string {
  const minutes = Math.max(0, Math.round((now - new Date(iso).getTime()) / 60000));
  if (minutes < 60) return `${minutes} мин`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours} ч ${minutes % 60} мин` : `${Math.floor(hours / 24)} дн`;
}
