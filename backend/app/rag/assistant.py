"""The RAG assistant: retrieve the right chunks, then let the LLM answer ONLY from them.

Person-aware retrieval: when a verified employee is selected, their profile is always put in the
context (exact lookup by id, not left to similarity search), and the policy search is run with
the question plus the employee's access level and status, so rules about that level come up.
"""
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from .documents import employee_profile_text
from .index import rag_index
from .llm import get_llm, provider_name

SYSTEM_PROMPT = """You are the Security & HR assistant of the Employee Verification Portal.
Rules:
1. Answer ONLY from the numbered sources in the context. Do not use outside knowledge.
2. Cite the sources you used in square brackets, like [S1] or [S2][S3], right after the fact.
3. If the sources do not contain the answer, say: "I couldn't find this in the company documents."
   and suggest who to contact if a source names one.
4. When asked whether an employee may enter a zone or do something, give a clear Yes / No /
   Needs approval first, then the reason, using both the employee profile and the policy.
5. Be short and practical: 2-6 sentences or a few bullet points. No preamble."""


def _format_sources(hits):
    sources, blocks = [], []
    for i, (doc, score) in enumerate(hits, start=1):
        sid = f"S{i}"
        md = doc.metadata
        blocks.append(f"[{sid}] ({md.get('title')} - {md.get('section') or 'General'})\n{doc.page_content}")
        sources.append({"id": sid, "title": md.get("title"), "section": md.get("section") or "",
                        "source": md.get("source"), "type": md.get("type"), "score": score,
                        "text": doc.page_content})
    return sources, "\n\n".join(blocks)


# Zone words -> minimum access level (Access Control Policy, section 3).
ZONE_RULES = [
    (("server room", "data centre", "data center", "datacenter"), "Server room (data centre)", 4),
    (("finance vault", "vault"), "Finance vault", 4),
    (("r&d lab", "r & d lab", "research lab", "lab"), "R&D lab", 3),
    (("network operations", "noc"), "Network operations room", 3),
    (("records room",), "Records room", 3),
    (("work floor", "office floor", "library", "training room"), "Office work floors / library / training rooms", 2),
    (("lobby", "cafeteria", "meeting room"), "Lobby / cafeteria / ground-floor meeting rooms", 1),
]


def access_check(question: str, employee):
    """Deterministic yes/no for 'can X enter <zone>' questions, so the decision never depends on
    the LLM's arithmetic. Returned as an extra cited source."""
    q = question.lower()
    for words, zone, need in ZONE_RULES:
        if any(re.search(rf"\b{re.escape(w.strip())}\b", q) for w in words):
            level = int(employee.access_level[1]) if employee.access_level[1:].isdigit() else 1
            if employee.status in ("Suspended", "Exited"):
                verdict = f"NOT ALLOWED - status is {employee.status}, so all access is blocked."
            elif employee.status == "On Leave" and need >= 3:
                verdict = "NOT ALLOWED - employees on leave cannot enter L3 or L4 zones."
            elif level >= need and employee.status == "On Leave":
                verdict = "NEEDS APPROVAL - on leave: office entry only with HR or manager approval."
            elif level >= need:
                verdict = "ALLOWED - access level is high enough."
            else:
                verdict = (f"NOT ALLOWED - needs L{need}, employee has {employee.access_level}. "
                           "A higher level can be requested via the HR portal (Access Control Policy, section 7).")
            text = (f"Access check for {employee.name}: zone '{zone}' requires L{need}. "
                    f"Employee access level {employee.access_level}, status {employee.status}. "
                    f"Decision: {verdict}")
            return text
    return None


def retrieve(question: str, employee=None, k: int | None = None):
    """Return [(Document, score)] for the question (and the selected employee, if any)."""
    k = k or settings.TOP_K
    if employee is None:
        return rag_index.search(question, k=k)
    from langchain_core.documents import Document
    profile = Document(page_content=employee_profile_text(employee), metadata={
        "type": "employee", "title": f"Employee profile - {employee.name}", "section": "Profile",
        "source": f"employee:{employee.employee_code}"})
    query = f"{question} access level {employee.access_level} status {employee.status}"
    hits = [(profile, 1.0)]
    check = access_check(question, employee)
    if check:
        hits.append((Document(page_content=check, metadata={
            "type": "rule", "title": "Access check (computed from Access Control Policy, section 3)",
            "section": "Zone rule", "source": "rule:access_check"}), 1.0))
    return hits + rag_index.search(query, k=k, doc_type="policy")


def _sentences(text: str) -> list[str]:
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*#|") for p in parts if len(p.strip()) > 25]


def extractive_answer(question: str, sources: list[dict], employee=None) -> str:
    """Used when no LLM is configured (or it fails): the most relevant lines, with citations."""
    q_words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}
    scored = []
    for src in sources:
        if src["type"] in ("employee", "rule"):
            continue
        for sent in _sentences(src["text"]):
            words = set(re.findall(r"[a-z0-9]+", sent.lower()))
            overlap = len(q_words & words)
            if employee is not None and (employee.access_level.lower() in words
                                         or employee.status.lower() in sent.lower()):
                overlap += 1
            if overlap:
                # Prefer sentences from higher-ranked sources.
                scored.append((overlap + 0.5 * src["score"], src["id"], sent.replace("**", "")))
    scored.sort(key=lambda t: -t[0])
    lines = []
    rule = next((s for s in sources if s["type"] == "rule"), None)
    if rule:
        lines.append(f"**{rule['text'].split('Decision: ')[-1]}** [{rule['id']}]")
    if employee is not None:
        lines.append(f"**{employee.name}** ({employee.employee_code}) - access level "
                     f"**{employee.access_level}**, status **{employee.status}** [S1].")
    seen = set()
    for _, sid, sent in scored:
        if sent not in seen:
            seen.add(sent)
            lines.append(f"- {sent} [{sid}]")
        if len(seen) == 4:
            break
    if len(lines) <= (1 if employee is not None else 0) + (1 if rule else 0):
        lines.append("I couldn't find this in the company documents.")
    return "\n".join(lines)


def answer(question: str, employee=None) -> dict:
    hits = retrieve(question, employee)
    sources, context = _format_sources(hits)
    llm = get_llm()
    mode, note = "llm", ""
    if llm is not None:
        user_msg = (f"Today's date: {date.today().isoformat()}\n\nContext:\n{context}\n\n"
                    f"Question: {question}")
        try:
            reply = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
        except Exception as exc:               # bad key, rate limit, no network ...
            mode, note = "extractive", f"LLM call failed ({type(exc).__name__}); showing matching policy lines."
            text = extractive_answer(question, sources, employee)
    else:
        mode, note = "extractive", "No LLM key configured; showing the most relevant policy lines."
        text = extractive_answer(question, sources, employee)
    return {"answer": text, "mode": mode, "provider": provider_name() if mode == "llm" else "none",
            "note": note, "sources": [{k: v for k, v in s.items() if k != "text"} | {"snippet": s["text"][:400]}
                                      for s in sources]}
