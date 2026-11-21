import { useState } from "react";

// Validated categorical slots 1-2 and the blue sequential ramp (dataviz reference palette).
const SERIES = [
  { key: "accuracy", name: "Train accuracy", color: "#2a78d6" },
  { key: "val_accuracy", name: "Validation accuracy", color: "#eb6834" },
];
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

export function AccuracyChart({ history }) {
  const [hover, setHover] = useState(null);
  const n = history?.accuracy?.length || 0;
  if (!n) return null;
  const W = 560, H = 240, m = { l: 40, r: 120, t: 12, b: 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (i) => m.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => m.t + (1 - v) * ih;
  const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  // End-of-line labels: keep at least 14px apart so they never overlap.
  const labelY = {};
  SERIES.filter((s) => history[s.key]).map((s) => ({ key: s.key, y: y(history[s.key][n - 1]) + 4 }))
    .sort((a, b) => a.y - b.y)
    .forEach((l, i, arr) => { labelY[l.key] = i && l.y - labelY[arr[i - 1].key] < 14 ? labelY[arr[i - 1].key] + 14 : l.y; });

  function move(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / iw) * (n - 1))));
    setHover(i);
  }

  return (
    <figure className="viz">
      <figcaption>Accuracy by epoch</figcaption>
      <div className="legend">
        {SERIES.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>)}
      </div>
      <div className="viz-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training and validation accuracy per epoch"
          onMouseMove={move} onMouseLeave={() => setHover(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={m.l + iw} y1={y(t)} y2={y(t)} className="grid" />
              <text x={m.l - 8} y={y(t) + 4} className="tick" textAnchor="end">{Math.round(t * 100)}%</text>
            </g>
          ))}
          <text x={m.l} y={H - 8} className="tick">epoch 1</text>
          <text x={m.l + iw} y={H - 8} className="tick" textAnchor="end">epoch {n}</text>
          {SERIES.map((s) => history[s.key] && (
            <g key={s.key}>
              <path d={path(history[s.key])} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
              <text x={x(n - 1) + 8} y={labelY[s.key]} className="direct-label">
                {s.name.split(" ")[0]} {Math.round(history[s.key][n - 1] * 100)}%
              </text>
            </g>
          ))}
          {hover != null && (
            <g>
              <line x1={x(hover)} x2={x(hover)} y1={m.t} y2={m.t + ih} className="crosshair" />
              {SERIES.map((s) => history[s.key] && (
                <circle key={s.key} cx={x(hover)} cy={y(history[s.key][hover])} r="4.5" fill={s.color} stroke="#fff" strokeWidth="2" />
              ))}
            </g>
          )}
        </svg>
        {hover != null && (
          <div className="tooltip" style={{ left: `${(x(hover) / W) * 100}%` }}>
            <strong>Epoch {hover + 1}</strong>
            {SERIES.map((s) => history[s.key] && (
              <div key={s.key}><i style={{ background: s.color }} />{s.name}: {(history[s.key][hover] * 100).toFixed(1)}%</div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}

export function ConfusionMatrix({ matrix, classes }) {
  const [hover, setHover] = useState(null);
  if (!matrix?.length) return null;
  const max = Math.max(1, ...matrix.flat());
  const color = (v) => (v === 0 ? "var(--surface-2)" : RAMP[Math.min(RAMP.length - 1, Math.floor((v / max) * (RAMP.length - 1)))]);
  const darkText = (v) => v / max > 0.45;
  return (
    <figure className="viz">
      <figcaption>Confusion matrix (validation images) - rows: true person, columns: predicted</figcaption>
      <div className="cm-wrap">
        <table className="cm">
          <thead>
            <tr><th></th>{classes.map((c) => <th key={c} scope="col"><span>{c}</span></th>)}</tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={classes[i]}>
                <th scope="row">{classes[i]}</th>
                {row.map((v, j) => (
                  <td key={j} style={{ background: color(v), color: darkText(v) ? "#fff" : "var(--text-primary)" }}
                    className={(i === j ? "diag " : "") + (hover && hover[0] === i && hover[1] === j ? "hover" : "")}
                    onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)}
                    title={`True ${classes[i]}, predicted ${classes[j]}: ${v}`}>
                    {v || ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        {hover ? `True ${classes[hover[0]]} → predicted ${classes[hover[1]]}: ${matrix[hover[0]][hover[1]]} image(s)` :
          "Diagonal = correct. Off-diagonal cells show which people the model confuses."}
      </p>
    </figure>
  );
}
