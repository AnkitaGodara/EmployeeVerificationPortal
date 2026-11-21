import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../App.jsx";
import ChatPanel from "../components/ChatPanel.jsx";
import { PageHeader } from "../components/ui.jsx";

export default function Assistant() {
  const { user } = useAuth();
  const [status, setStatus] = useState(null);
  const [employees, setEmployees] = useState([]);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    api.get("/api/assistant/status").then(setStatus).catch(() => {});
    if (user.role === "admin") api.get("/api/employees").then(setEmployees).catch(() => {});
  }, [user.role]);

  const employee = employees.find((e) => String(e.id) === selected) || null;
  return (
    <>
      <PageHeader title="Policy assistant (RAG)"
        subtitle="Answers are generated only from the company documents and employee records, with numbered sources." />
      {status && (
        <div className="pipeline card">
          <div><span className="label">Retriever</span>{status.index?.retrieval === "hybrid" ? "Hybrid: FAISS vectors + BM25, fused with RRF" : status.index?.retrieval} · {status.embeddings} embeddings · top {status.top_k}</div>
          <div><span className="label">Knowledge base</span>{status.index?.policies?.length ?? 0} policy docs · {status.index?.employees ?? 0} employee profiles · {status.index?.chunks ?? 0} chunks</div>
          <div><span className="label">Generator</span>{status.llm_provider === "none" ? "No LLM key - extractive answers" : `LLM: ${status.llm_provider}`}</div>
        </div>
      )}
      <div className="card">
        {user.role === "admin" && (
          <label className="inline">
            Ask about a specific employee (optional)
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              <option value="">No employee - general policy question</option>
              {employees.map((e) => <option key={e.id} value={e.id}>{e.name} ({e.employee_code}, {e.access_level}, {e.status})</option>)}
            </select>
          </label>
        )}
        <ChatPanel employee={employee} suggestions={employee ? [
          `Can ${employee.name} enter the R&D lab?`,
          `Is ${employee.name} allowed into the office today?`,
          `How can ${employee.name} get L4 access?`,
        ] : [
          "What does access level L3 allow?",
          "What should security do if the result is Unknown?",
          "How many sick leaves do employees get?",
          "How long are verification logs kept?",
          "Can a visitor enter the server room?",
        ]} />
      </div>
    </>
  );
}
