import type {
  AuditTrail,
  Escalation,
  EscalationContext,
  KbDocument,
  KbVersion,
  QueuePage,
  ResolveAction,
} from "./types";

/** Ошибка в едином формате API: {"error": {"code", "message", "trace_id"}}. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly traceId?: string,
  ) {
    super(message);
  }
}

export class Api {
  constructor(
    private readonly token: string,
    private readonly onUnauthorized: () => void,
  ) {}

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await fetch(path, {
      method,
      headers: {
        Authorization: `Bearer ${this.token}`,
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });

    if (response.status === 401) this.onUnauthorized();
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const error = payload?.error;
      throw new ApiError(
        response.status,
        error?.code ?? "http_error",
        error?.message ?? `HTTP ${response.status}`,
        error?.trace_id,
      );
    }
    return (await response.json()) as T;
  }

  // --- эскалации ---
  queue = (cursor?: string | null) =>
    this.request<QueuePage>(
      "GET",
      `/api/v1/escalations?limit=50${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`,
    );
  context = (id: string) => this.request<EscalationContext>("GET", `/api/v1/escalations/${id}`);
  claim = (id: string) => this.request<Escalation>("POST", `/api/v1/escalations/${id}/claim`);
  release = (id: string) => this.request<Escalation>("DELETE", `/api/v1/escalations/${id}/claim`);
  resolve = (id: string, action: ResolveAction, finalText?: string) =>
    this.request<{ final_text: string }>("POST", `/api/v1/escalations/${id}/resolve`, {
      action,
      final_text: finalText,
    });
  audit = (ticketId: string) => this.request<AuditTrail>("GET", `/api/v1/tickets/${ticketId}/audit`);

  // --- база знаний ---
  kbList = () => this.request<{ items: KbDocument[] }>("GET", "/api/v1/kb/documents?limit=200");
  kbCreate = (doc: { slug: string; title: string; content: string }) =>
    this.request<KbDocument>("POST", "/api/v1/kb/documents", doc);
  kbUpdate = (id: string, patch: { title?: string; content?: string }) =>
    this.request<KbDocument>("PUT", `/api/v1/kb/documents/${id}`, patch);
  kbDelete = (id: string) => this.request<{ id: string }>("DELETE", `/api/v1/kb/documents/${id}`);
  kbVersions = (id: string) =>
    this.request<{ items: KbVersion[] }>("GET", `/api/v1/kb/documents/${id}/versions`);
}
