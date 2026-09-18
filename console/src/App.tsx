import { useCallback, useEffect, useMemo, useState } from "react";
import { Api } from "./api";
import { EscalationView } from "./pages/EscalationView";
import { KnowledgeBase } from "./pages/KnowledgeBase";
import { Login } from "./pages/Login";
import { Queue } from "./pages/Queue";
import { loadSession, saveSession, type Session } from "./session";
import { useEscalationFeed, type FeedEvent } from "./useEscalationFeed";

type Route = { page: "queue" } | { page: "escalation"; id: string } | { page: "kb" };

function parseHash(hash: string): Route {
  const escalation = hash.match(/^#\/escalations\/([0-9a-f-]{36})$/);
  if (escalation) return { page: "escalation", id: escalation[1] };
  if (hash === "#/kb") return { page: "kb" };
  return { page: "queue" };
}

export function App() {
  const [session, setSession] = useState<Session | null>(loadSession);

  const signOut = useCallback(() => {
    saveSession(null);
    setSession(null);
  }, []);

  if (!session) {
    return (
      <Login
        onSignIn={(next) => {
          saveSession(next);
          setSession(next);
        }}
      />
    );
  }
  return <Shell session={session} onSignOut={signOut} />;
}

function Shell({ session, onSignOut }: { session: Session; onSignOut: () => void }) {
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
  const [queueVersion, setQueueVersion] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const api = useMemo(() => new Api(session.token, onSignOut), [session.token, onSignOut]);

  useEffect(() => {
    const onHash = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const onFeed = useCallback((event: FeedEvent) => {
    setQueueVersion((value) => value + 1);
    if (event.type === "escalation.queued") {
      setNotice(event.priority && event.priority > 0 ? "Новая приоритетная эскалация" : "Новая эскалация");
      window.setTimeout(() => setNotice(null), 4000);
    }
  }, []);
  const feed = useEscalationFeed(session.token, onFeed);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">Консоль поддержки</div>
        <nav>
          <a className={route.page !== "kb" ? "active" : ""} href="#/queue">
            Очередь
          </a>
          {session.role === "admin" && (
            <a className={route.page === "kb" ? "active" : ""} href="#/kb">
              База знаний
            </a>
          )}
        </nav>
        <div className="topbar-right">
          <span className={`feed feed-${feed}`} title="Push-уведомления об эскалациях">
            {feed === "live" ? "онлайн" : feed === "connecting" ? "подключение…" : "офлайн, обновление по таймеру"}
          </span>
          <span className="role">{session.role === "admin" ? "администратор" : "оператор"}</span>
          <button className="link" onClick={onSignOut}>
            Выйти
          </button>
        </div>
      </header>

      {notice && (
        <div className="notice" role="status">
          {notice}
        </div>
      )}

      <main>
        {route.page === "queue" && <Queue api={api} version={queueVersion} feedLive={feed === "live"} />}
        {route.page === "escalation" && (
          <EscalationView api={api} id={route.id} operatorId={session.operatorId} />
        )}
        {route.page === "kb" && session.role === "admin" && <KnowledgeBase api={api} />}
      </main>
    </div>
  );
}
