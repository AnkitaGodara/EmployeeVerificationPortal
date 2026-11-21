import { AuthImage, Badge, LevelChip } from "./ui.jsx";

function ConfidenceBar({ value, threshold }) {
  const pct = Math.round(value * 100);
  return (
    <div className="conf">
      <div className="conf-track" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
        aria-label="Model confidence">
        <div className={`conf-fill ${value >= threshold ? "ok" : "low"}`} style={{ width: `${pct}%` }} />
        <div className="conf-threshold" style={{ left: `${threshold * 100}%` }} title={`Threshold ${threshold * 100}%`} />
      </div>
      <span className="conf-label">{pct}% confidence <span className="muted">(threshold {Math.round(threshold * 100)}%)</span></span>
    </div>
  );
}

const FIELDS = [
  ["Employee code", "employee_code"], ["Department", "department"], ["Designation", "designation"],
  ["Shift", "shift"], ["Office", "office_location"], ["Manager", "manager"],
  ["Joined", "joining_date"], ["Email", "email"],
];

export default function ResultCard({ result }) {
  if (!result) {
    return (
      <div className="card result empty">
        <p className="muted">The verification result will appear here.</p>
      </div>
    );
  }
  const emp = result.employee;
  return (
    <div className={`card result ${result.result.toLowerCase()}`}>
      <div className="result-head">
        <Badge value={result.result} />
        <span>{result.message}</span>
      </div>
      {result.result !== "NO_FACE" && <ConfidenceBar value={result.confidence} threshold={result.threshold} />}
      {result.warning && <div className="notice error">{result.warning}</div>}
      {emp && (
        <div className="emp">
          <AuthImage src={`/api/employees/${emp.id}/photo`} alt={emp.name} className="emp-photo"
            fallback={emp.name.slice(0, 1)} />
          <div className="emp-main">
            <h2>{emp.name}</h2>
            <div className="emp-tags">
              <LevelChip level={emp.access_level} />
              <Badge value={emp.status} />
            </div>
            <dl className="fields">
              {FIELDS.map(([label, key]) => (
                <div key={key}><dt>{label}</dt><dd>{emp[key] || "—"}</dd></div>
              ))}
            </dl>
          </div>
        </div>
      )}
      {emp && (
        <div className="zones">
          <div className="label">Allowed zones</div>
          {result.allowed_zones.length ? (
            <div className="zone-list">{result.allowed_zones.map((z) => <span key={z} className="zone">{z}</span>)}</div>
          ) : (
            <span className="muted">No zones - access blocked.</span>
          )}
        </div>
      )}
      {result.top?.length > 1 && (
        <details className="candidates">
          <summary>Top {result.top.length} model candidates</summary>
          <ul>
            {result.top.map((t) => (
              <li key={t.label}><span>{t.name}</span><span className="num">{(t.confidence * 100).toFixed(1)}%</span></li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
