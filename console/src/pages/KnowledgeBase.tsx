import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { ApiError, type Api } from "../api";
import { formatTime } from "../labels";
import type { KbDocument, KbVersion } from "../types";

// Индексация идёт в фоне: пока у документа есть непроиндексированная правка,
// список перечитывается, чтобы метка «индексируется» ушла сама.
const INDEXING_POLL_MS = 2000;

export function KnowledgeBase({ api }: { api: Api }) {
  const [docs, setDocs] = useState<KbDocument[] | null>(null);
  const [selected, setSelected] = useState<string | "new" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  // Выбор документа внизу длинного списка не должен оставлять редактор за экраном.
  useEffect(() => {
    if (selected) editorRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);

  const reload = useCallback(() => {
    api
      .kbList()
      .then((page) => (setDocs(page.items), setError(null)))
      .catch((err: Error) => setError(err.message));
  }, [api]);

  useEffect(reload, [reload]);

  useEffect(() => {
    if (!docs?.some((doc) => doc.indexing)) return;
    const timer = window.setTimeout(reload, INDEXING_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [docs, reload]);

  if (error) return <p className="error">{error}</p>;
  if (!docs) return <p className="muted">Загрузка…</p>;

  const current = docs.find((doc) => doc.id === selected) ?? null;

  return (
    <section>
      <div className="section-head">
        <h2>База знаний</h2>
        <button className="primary" onClick={() => setSelected("new")}>
          Новый документ
        </button>
      </div>
      <div className="columns">
        <ul className="kb-list column-side">
          {docs.map((doc) => (
            <li key={doc.id}>
              <button
                className={`kb-item ${doc.id === selected ? "active" : ""}`}
                onClick={() => setSelected(doc.id)}
              >
                <span>{doc.title}</span>
                <span className="muted small">
                  {doc.slug} · v{doc.latest_version}
                  {doc.indexing && <span className="tag tag-warn">индексируется</span>}
                </span>
              </button>
            </li>
          ))}
        </ul>
        <div className="column-main" ref={editorRef}>
          {selected === "new" && (
            <Editor
              key="new"
              api={api}
              onSaved={(doc) => {
                setSelected(doc.id);
                reload();
              }}
            />
          )}
          {current && (
            <Editor
              key={current.id}
              api={api}
              doc={current}
              onSaved={reload}
              onDeleted={() => {
                setSelected(null);
                reload();
              }}
            />
          )}
          {!selected && <div className="empty card">Выберите документ или создайте новый.</div>}
        </div>
      </div>
    </section>
  );
}

function Editor({
  api,
  doc,
  onSaved,
  onDeleted,
}: {
  api: Api;
  doc?: KbDocument;
  onSaved: (doc: KbDocument) => void;
  onDeleted?: () => void;
}) {
  const [slug, setSlug] = useState(doc?.slug ?? "");
  const [title, setTitle] = useState(doc?.title ?? "");
  const [content, setContent] = useState(doc?.content ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [versions, setVersions] = useState<KbVersion[] | null>(null);

  useEffect(() => {
    if (doc) api.kbVersions(doc.id).then((page) => setVersions(page.items)).catch(() => setVersions([]));
  }, [api, doc]);

  const dirty = !doc || title !== doc.title || content !== doc.content;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const saved = doc
        ? await api.kbUpdate(doc.id, {
            ...(title !== doc.title ? { title } : {}),
            ...(content !== doc.content ? { content } : {}),
          })
        : await api.kbCreate({ slug, title, content });
      onSaved(saved);
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message} (${err.code})` : String(err));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!doc || !confirm(`Удалить «${doc.title}»? Документ уйдёт из ответов агента, история сохранится.`)) return;
    setBusy(true);
    try {
      await api.kbDelete(doc.id);
      onDeleted?.();
    } catch (err) {
      setError(String(err));
      setBusy(false);
    }
  };

  return (
    <form className="card editor" onSubmit={submit}>
      {doc?.indexing && (
        <p className="tag tag-warn">
          Правка v{doc.latest_version} индексируется. Агент пока отвечает по v{doc.current_version ?? "—"}.
        </p>
      )}
      <label htmlFor="slug">Slug</label>
      <input
        id="slug"
        value={slug}
        onChange={(event) => setSlug(event.target.value)}
        disabled={Boolean(doc)}
        pattern="[a-z0-9][a-z0-9-]{1,126}"
        placeholder="delivery-terms"
        required
      />
      <label htmlFor="title">Заголовок</label>
      <input id="title" value={title} onChange={(event) => setTitle(event.target.value)} required />
      <label htmlFor="content">Текст</label>
      <textarea id="content" rows={8} value={content} onChange={(event) => setContent(event.target.value)} required />
      {error && <p className="error">{error}</p>}
      <div className="buttons">
        <button className="primary" type="submit" disabled={busy || !dirty}>
          {doc ? "Сохранить новую версию" : "Создать"}
        </button>
        {doc && (
          <button type="button" className="danger" disabled={busy} onClick={remove}>
            Удалить
          </button>
        )}
      </div>

      {doc && versions && versions.length > 0 && (
        <div className="versions">
          <h3>История версий</h3>
          <ol>
            {[...versions].reverse().map((version) => (
              <li key={version.id}>
                <details>
                  <summary>
                    v{version.version} · {formatTime(version.created_at)}
                    {version.version === doc.current_version && <span className="tag">текущая</span>}
                    {!version.indexed && <span className="tag tag-warn">не проиндексирована</span>}
                  </summary>
                  <p className="small">{version.content}</p>
                </details>
              </li>
            ))}
          </ol>
        </div>
      )}
    </form>
  );
}
