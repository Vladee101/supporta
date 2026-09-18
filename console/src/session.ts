import type { Role } from "./types";

const STORAGE_KEY = "support-console-token";

export interface Session {
  token: string;
  operatorId: string;
  role: Role;
  expiresAt: Date;
}

/**
 * Читает payload токена, не проверяя подпись: проверяет её сервер на каждом
 * запросе. Здесь нужно только понять роль (показывать ли базу знаний) и срок
 * жизни (не отправлять заведомо протухший токен).
 */
export function parseToken(token: string): Session | null {
  try {
    const [payload] = token.trim().split(".");
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    const claims = JSON.parse(json) as { typ?: string; sub: string; role: Role; exp: number };
    if (claims.typ !== "operator") return null;
    return {
      token: token.trim(),
      operatorId: claims.sub,
      role: claims.role,
      expiresAt: new Date(claims.exp * 1000),
    };
  } catch {
    return null;
  }
}

export function loadSession(): Session | null {
  try {
    const token = sessionStorage.getItem(STORAGE_KEY);
    const session = token ? parseToken(token) : null;
    return session && session.expiresAt > new Date() ? session : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session | null): void {
  try {
    if (session) sessionStorage.setItem(STORAGE_KEY, session.token);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Хранилище недоступно (приватный режим) - сессия живёт до перезагрузки.
  }
}
