// Контракты ответов API. Соответствуют app/escalations/operations.py,
// app/api/audit.py и app/kb/service.py.

export type Role = "operator" | "admin";

export interface Escalation {
  id: string;
  ticket_id: string;
  reason: string;
  rule_id: string | null;
  status: "pending" | "in_progress" | "resolved";
  priority: number;
  draft_text: string | null;
  locked_by: string | null;
  locked_until: string | null;
  created_at: string;
  resolved_at: string | null;
}

export interface QueuePage {
  items: Escalation[];
  next_cursor: string | null;
}

export interface ContextMessage {
  sender: "client" | "agent" | "operator";
  iteration: number;
  content: string;
  created_at: string;
}

export interface ContextClassification {
  iteration: number;
  category: string;
  confidence: number | null;
  confidence_source: string | null;
  model_id: string | null;
  reasoning: string | null;
}

export interface ContextDocument {
  iteration: number;
  rank: number;
  relevance_score: number;
  document_version_id: string;
  slug: string;
  title: string;
  version: number;
  snapshot: string | null;
}

export interface EscalationContext {
  escalation: Escalation;
  ticket: {
    id: string;
    channel: string;
    status: string;
    category: string | null;
    risk_level: string | null;
    clarification_count: number;
    created_at: string;
  };
  messages: ContextMessage[];
  classifications: ContextClassification[];
  documents: ContextDocument[];
}

export interface AuditEvent {
  at: string;
  actor: string;
  action: string;
  rule_id: string | null;
  class_confidence: number | null;
  rag_confidence: number | null;
  reasoning: string | null;
  payload: Record<string, unknown> | null;
  trace_id: string | null;
}

export interface AuditTrail {
  ticket: { id: string; status: string; category: string | null; clarification_count: number };
  clarification_limit_reached: boolean;
  classifications: (ContextClassification & { created_at: string })[];
  events: AuditEvent[];
}

export interface KbDocument {
  id: string;
  slug: string;
  title: string;
  content: string;
  latest_version: number;
  current_version: number | null;
  indexing: boolean;
  embedding_model: string | null;
  updated_at: string;
}

export interface KbVersion {
  id: string;
  version: number;
  content: string;
  indexed: boolean;
  embedding_model: string | null;
  created_at: string;
}

export type ResolveAction = "confirm" | "edit" | "reject";
