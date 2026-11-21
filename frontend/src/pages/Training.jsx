import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { AccuracyChart, ConfusionMatrix } from "../components/TrainingCharts.jsx";
import { Badge, Notice, PageHeader, Spinner, formatDate } from "../components/ui.jsx";

function ZipUpload({ onDone }) {
  const ref = useRef(null);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  async function upload(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    setReport(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await api.upload("/api/enroll/zip", form);
      setReport(r);
      onDone();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      ref.current.value = "";
    }
  }

  return (
    <div className="card">
      <h2 className="section-title">1. Upload photos</h2>
      <p className="muted">
        Upload one <strong>.zip</strong> with a folder per person (e.g. <code>project 1/harshal/*.jpg</code>,{" "}
        <code>project 1/kishan/*.jpg</code>). Folder names become employees. Each photo is face-detected,
        cropped to 100×100 and saved. To add photos for one person, use <em>Upload folder</em> on the Employees page.
      </p>
      <input ref={ref} type="file" accept=".zip" hidden onChange={(e) => upload(e.target.files?.[0])} />
      <button className="btn primary" disabled={busy} onClick={() => ref.current.click()}>
        {busy ? <><Spinner /> Processing faces…</> : "Upload zip of person folders"}
      </button>
      <Notice kind="error">{error}</Notice>
      {report && (
        <div className="report">
          <Notice kind={report.total_skipped ? "warning" : "info"}>
            {report.total_saved} faces saved, {report.total_skipped} images skipped across {report.people.length} people.
          </Notice>
          <table>
            <thead><tr><th>Person</th><th></th><th className="num">Saved</th><th className="num">Skipped</th><th className="num">Total faces</th></tr></thead>
            <tbody>
              {report.people.map((p) => (
                <tr key={p.label}>
                  <td>{p.name} <span className="muted small">({p.label})</span></td>
                  <td>{p.created ? <span className="badge info">new employee</span> : ""}</td>
                  <td className="num">{p.saved}</td>
                  <td className="num" title={p.skipped_files.map((f) => `${f.file}: ${f.reason}`).join("\n")}>{p.skipped}</td>
                  <td className="num">{p.total_faces}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-value">{value == null ? "—" : `${(value * 100).toFixed(1)}%`}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export default function Training() {
  const [employees, setEmployees] = useState([]);
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [runs, setRuns] = useState([]);
  const [epochs, setEpochs] = useState(40);
  const [error, setError] = useState("");

  const loadAll = () => {
    api.get("/api/employees").then(setEmployees).catch(() => {});
    api.get("/api/train/status").then(setStatus).catch(() => {});
    api.get("/api/train/metrics").then(setMetrics).catch(() => setMetrics(null));
    api.get("/api/train/runs").then(setRuns).catch(() => {});
  };
  useEffect(() => { loadAll(); }, []);

  useEffect(() => {
    if (status?.state !== "running") return undefined;
    const t = setInterval(async () => {
      const s = await api.get("/api/train/status").catch(() => null);
      if (!s) return;
      setStatus(s);
      if (s.state !== "running") loadAll();
    }, 1500);
    return () => clearInterval(t);
  }, [status?.state]);

  async function train() {
    setError("");
    try {
      const r = await api.post(`/api/train?epochs=${epochs}`);
      setStatus({ ...r.status, model_ready: status?.model_ready, classes: status?.classes || [] });
    } catch (e) {
      setError(e.message);
    }
  }

  const ready = employees.filter((e) => e.face_images >= 5);
  const running = status?.state === "running";
  return (
    <>
      <PageHeader title="Enrol & train" subtitle="Upload face photos, train the CNN, and check how well it recognises each person." />
      <ZipUpload onDone={loadAll} />

      <div className="card">
        <h2 className="section-title">2. Train the model</h2>
        <div className="face-counts">
          {employees.map((e) => (
            <span key={e.id} className={`count-chip ${e.face_images >= 10 ? "" : e.face_images >= 5 ? "warn" : "bad"}`}>
              {e.name}: {e.face_images}
            </span>
          ))}
        </div>
        <p className="muted small">
          {ready.length} of {employees.length} people have 5+ face images (needed). 10+ per person with different angles and lighting is recommended.
          Training uses an 80/20 split, data augmentation and early stopping.
        </p>
        <div className="row">
          <label className="inline">Epochs
            <input type="number" min="1" max="200" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} style={{ width: 90 }} />
          </label>
          <button className="btn primary" disabled={running || ready.length < 2} onClick={train}>
            {running ? <><Spinner /> Training…</> : "Train model"}
          </button>
        </div>
        {running && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${(status.epoch / Math.max(1, status.epochs)) * 100}%` }} />
            <span>{status.message}</span>
          </div>
        )}
        {status && !running && status.message && <Notice kind={status.state === "failed" ? "error" : "info"}>{status.message}</Notice>}
        <Notice kind="error">{error}</Notice>
      </div>

      {metrics && (
        <div className="card">
          <h2 className="section-title">3. Model evaluation <span className="muted small">trained {formatDate(metrics.trained_at)}</span></h2>
          <div className="stat-row">
            <Metric label="Accuracy" value={metrics.accuracy} />
            <Metric label="Precision (macro)" value={metrics.precision_macro} />
            <Metric label="Recall (macro)" value={metrics.recall_macro} />
            <Metric label="F1 (macro)" value={metrics.f1_macro} />
          </div>
          <p className="muted small">
            {metrics.num_images} images of {metrics.classes.length} people · {metrics.train_images} train / {metrics.val_images} validation ·
            {" "}{metrics.epochs_run} epochs (early stopping) · Unknown threshold {Math.round(metrics.confidence_threshold * 100)}%.
            {metrics.val_images < 30 && " With few validation images these scores move a lot between runs."}
          </p>
          <div className="grid-2">
            <AccuracyChart history={metrics.history} />
            <ConfusionMatrix matrix={metrics.confusion_matrix} classes={metrics.classes} />
          </div>
          <details>
            <summary>Per-person scores</summary>
            <table>
              <thead><tr><th>Person</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Val images</th></tr></thead>
              <tbody>
                {metrics.classes.map((c) => {
                  const r = metrics.per_class[c];
                  return <tr key={c}><td>{c}</td><td className="num">{r.precision.toFixed(2)}</td><td className="num">{r.recall.toFixed(2)}</td><td className="num">{r["f1-score"].toFixed(2)}</td><td className="num">{r.support}</td></tr>;
                })}
              </tbody>
            </table>
          </details>
        </div>
      )}

      {runs.length > 0 && (
        <div className="card table-card">
          <h2 className="section-title pad">Training history</h2>
          <table>
            <thead><tr><th>#</th><th>Started</th><th>By</th><th>Status</th><th className="num">Images</th><th className="num">People</th><th className="num">Accuracy</th><th className="num">F1</th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td><td>{formatDate(r.started_at)}</td><td>{r.started_by}</td><td><Badge value={r.status} /></td>
                  <td className="num">{r.num_images}</td><td className="num">{r.num_classes}</td>
                  <td className="num">{r.accuracy == null ? "—" : `${(r.accuracy * 100).toFixed(1)}%`}</td>
                  <td className="num">{r.macro_f1 == null ? "—" : r.macro_f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
