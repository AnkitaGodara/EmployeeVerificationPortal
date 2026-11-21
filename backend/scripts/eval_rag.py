"""Evaluate retrieval quality on a labelled question set (eval/rag_eval_set.json).

For each question we know which policy section(s) hold the answer. We check whether the retriever
returns that section, and where:
  Hit@1  - the right section is the first result
  Hit@k  - the right section is anywhere in the top k (what the LLM actually sees)
  MRR    - mean of 1/rank of the right section (1.0 = always first)
If an LLM key is configured (--answers), it also asks every question end to end and checks that
the answer contains the expected key fact and cites at least one source.

Run from the backend folder:  python -m scripts.eval_rag   [--k 4] [--answers]
"""
import argparse
import json
from datetime import date
from pathlib import Path

from app.config import settings
from app.db import init_db
from app.rag.index import rag_index

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def is_relevant(doc, item) -> bool:
    """A chunk is relevant if it comes from any of the accepted (file, section) pairs."""
    return any(doc.metadata.get("source") == src and doc.metadata.get("section", "").startswith(sec)
               for src, sec in item["accept"])


def evaluate(items, mode, k):
    hit1 = hitk = rr = 0.0
    misses = []
    for item in items:
        hits = rag_index.search(item["q"], k=k, doc_type="policy", mode=mode)
        rank = next((i + 1 for i, (doc, _) in enumerate(hits) if is_relevant(doc, item)), None)
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        if rank is None:
            misses.append(item["q"])
    n = len(items)
    return {"mode": mode, "hit@1": round(hit1 / n, 3), f"hit@{k}": round(hitk / n, 3),
            "mrr": round(rr / n, 3), "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=settings.TOP_K)
    ap.add_argument("--answers", action="store_true", help="also test full answers (needs an LLM key)")
    args = ap.parse_args()

    init_db()
    rag_index.build()
    items = json.loads((EVAL_DIR / "rag_eval_set.json").read_text())
    results = [evaluate(items, m, args.k) for m in ("vector", "bm25", "hybrid")]

    lines = [f"# RAG retrieval evaluation ({date.today().isoformat()})", "",
             f"{len(items)} labelled questions · {rag_index.meta['chunks']} chunks · "
             f"embeddings: `{settings.EMBEDDINGS_PROVIDER}` · k = {args.k}", "",
             f"| Retriever | Hit@1 | Hit@{args.k} | MRR |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['mode']} | {r['hit@1']:.0%} | {r[f'hit@{args.k}']:.0%} | {r['mrr']:.2f} |")
    for r in results:
        if r["misses"]:
            lines += ["", f"**{r['mode']} missed:** " + "; ".join(r["misses"])]

    if args.answers:
        from app.rag.assistant import answer
        from app.rag.llm import provider_name
        if provider_name() == "none":
            lines += ["", "_Answer evaluation skipped: no LLM key configured._"]
        else:
            ok = cited = 0
            for item in items:
                a = answer(item["q"])
                text = a["answer"].lower()
                ok += all(kw.lower() in text for kw in item["keywords"])
                cited += "[s" in text
            n = len(items)
            lines += ["", f"## End-to-end answers ({provider_name()})", "",
                      f"- Contains the expected key fact: {ok}/{n} ({ok / n:.0%})",
                      f"- Cites at least one source: {cited}/{n} ({cited / n:.0%})"]

    report = "\n".join(lines) + "\n"
    (EVAL_DIR / "results.md").write_text(report)
    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(report)


if __name__ == "__main__":
    main()
