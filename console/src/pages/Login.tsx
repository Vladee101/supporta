import { useState, type FormEvent } from "react";
import { parseToken, type Session } from "../session";

export function Login({ onSignIn }: { onSignIn: (session: Session) => void }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const session = parseToken(token);
    if (!session) {
      setError("Это не токен оператора. Токен тикета клиента здесь не подойдёт.");
      return;
    }
    if (session.expiresAt <= new Date()) {
      setError("Срок действия токена истёк - выпустите новый.");
      return;
    }
    onSignIn(session);
  };

  return (
    <div className="login">
      <form className="card login-card" onSubmit={submit}>
        <h1>Консоль поддержки</h1>
        <p className="muted">
          Вставьте токен оператора. Выпустить его можно командой
          <code>python -m scripts.issue_token --email you@example.com</code>
        </p>
        <label htmlFor="token">Токен</label>
        <textarea
          id="token"
          rows={4}
          value={token}
          onChange={(event) => {
            setToken(event.target.value);
            setError(null);
          }}
          placeholder="eyJ0eXAiOiJvcGVyYXRvciIs…"
          spellCheck={false}
        />
        {error && <p className="error">{error}</p>}
        <button className="primary" type="submit" disabled={!token.trim()}>
          Войти
        </button>
      </form>
    </div>
  );
}
