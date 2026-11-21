import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../App.jsx";
import Markdown from "../components/Markdown.jsx";
import { Notice, PageHeader, Spinner } from "../components/ui.jsx";

export default function Policies() {
  const { user } = useAuth();
  const isAdmin = user.role === "admin";
  const [docs, setDocs] = useState([]);
  const [open, setOpen] = useState(null);
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const ref = useRef(null);

  const load = () => {
    api.get("/api/documents").then((d) => {
      setDocs(d);
      if (!open && d.length) view(d[0].name);
    }).catch(() => {});
    api.get("/api/assistant/status").then(setStatus).catch(() => {});
  };
  useEffect(() => { load(); }, []);

  async function view(name) {
    setOpen(await api.get(`/api/documents/${encodeURIComponent(name)}`));
  }

  async function upload(file) {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await api.upload("/api/documents", form);
      setNotice({ kind: "info", text: `${r.name} added. Index rebuilt: ${r.index.chunks} chunks.` });
      load();
      view(r.name);
    } catch (e) {
      setNotice({ kind: "error", text: e.message });
    } finally {
      setBusy(false);
      ref.current.value = "";
    }
  }

  async function remove(name) {
    if (!window.confirm(`Remove ${name} from the knowledge base?`)) return;
    await api.del(`/api/documents/${encodeURIComponent(name)}`).catch((e) => setNotice({ kind: "error", text: e.message }));
    setOpen(null);
    load();
  }

  async function reindex() {
    setBusy(true);
    try {
      const meta = await api.post("/api/documents/reindex");
      setNotice({ kind: "info", text: `Index rebuilt: ${meta.documents} documents, ${meta.chunks} chunks.` });
      load();
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Company policies" subtitle="The knowledge base the RAG assistant searches. Employee profiles are added automatically.">
        {isAdmin && (
          <>
            <input ref={ref} type="file" hidden accept=".md,.txt,.pdf" onChange={(e) => upload(e.target.files?.[0])} />
            <button className="btn ghost" disabled={busy} onClick={reindex}>Rebuild index</button>
            <button className="btn primary" disabled={busy} onClick={() => ref.current.click()}>
              {busy ? <Spinner /> : null} Upload document
            </button>
          </>
        )}
      </PageHeader>
      {notice && <Notice kind={notice.kind} onClose={() => setNotice(null)}>{notice.text}</Notice>}
      {status?.index && (
        <p className="muted small">
          Index: {status.index.documents} documents → {status.index.chunks} chunks · embeddings: {status.index.embeddings} · vector store: FAISS
        </p>
      )}
      <div className="docs-layout">
        <ul className="doc-list card">
          {docs.map((d) => (
            <li key={d.name}>
              <button className={open?.name === d.name ? "doc active" : "doc"} onClick={() => view(d.name)}>
                <strong>{d.title}</strong>
                <span className="muted small">{d.name} · {d.origin}</span>
              </button>
            </li>
          ))}
        </ul>
        <article className="card doc-view">
          {open ? (
            <>
              {isAdmin && docs.find((d) => d.name === open.name)?.origin === "uploaded" && (
                <button className="btn small danger float-right" onClick={() => remove(open.name)}>Remove</button>
              )}
              <Markdown text={open.content} />
            </>
          ) : <p className="muted">Choose a document.</p>}
        </article>
      </div>
    </>
  );
}
