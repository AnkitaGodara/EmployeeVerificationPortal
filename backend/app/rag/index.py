"""The knowledge-base index: FAISS vectors + BM25 keywords over the same chunks.

search() supports three modes (RETRIEVAL_MODE): vector, bm25, or hybrid (both, fused with
Reciprocal Rank Fusion). Hybrid is the default; scripts/eval_rag.py compares all three.
"""
import json
import threading

from langchain_community.vectorstores import FAISS

from ..config import settings
from ..db import Employee, SessionLocal
from .bm25 import BM25, rrf
from .documents import load_employee_documents, load_policy_documents, split_documents
from .embeddings import make_embeddings

META_FILE = settings.INDEX_DIR / "index_meta.json"


class RagIndex:
    def __init__(self):
        self._lock = threading.Lock()
        self.store: FAISS | None = None
        self.chunks: list = []
        self.bm25: BM25 | None = None
        self.meta: dict = {}

    def _activate(self, store: FAISS, meta: dict):
        ids = store.index_to_docstore_id
        chunks = [store.docstore.search(ids[i]) for i in range(len(ids))]
        bm25 = BM25([c.page_content for c in chunks])
        with self._lock:
            self.store, self.chunks, self.bm25, self.meta = store, chunks, bm25, meta

    def build(self) -> dict:
        """(Re)build the whole index. The corpus is small, so a full rebuild takes about a second."""
        db = SessionLocal()
        try:
            employees = db.query(Employee).all()
            docs = load_policy_documents() + load_employee_documents(employees)
        finally:
            db.close()
        chunks = split_documents(docs)
        provider = settings.EMBEDDINGS_PROVIDER
        emb = make_embeddings(provider, corpus=[c.page_content for c in chunks], folder=settings.INDEX_DIR)
        store = FAISS.from_documents(chunks, emb)
        store.save_local(str(settings.INDEX_DIR))
        meta = {"embeddings": provider, "retrieval": settings.RETRIEVAL_MODE, "documents": len(docs),
                "chunks": len(chunks),
                "policies": sorted({d.metadata["source"] for d in docs if d.metadata["type"] == "policy"}),
                "employees": len(employees)}
        META_FILE.write_text(json.dumps(meta, indent=2))
        self._activate(store, meta)
        return meta

    def ensure(self):
        if self.store is not None:
            return
        if (settings.INDEX_DIR / "index.faiss").exists() and META_FILE.exists():
            meta = json.loads(META_FILE.read_text())
            if meta.get("embeddings") == settings.EMBEDDINGS_PROVIDER:
                emb = make_embeddings(settings.EMBEDDINGS_PROVIDER, folder=settings.INDEX_DIR)
                store = FAISS.load_local(str(settings.INDEX_DIR), emb,
                                         allow_dangerous_deserialization=True)  # our own files only
                meta["retrieval"] = settings.RETRIEVAL_MODE
                self._activate(store, meta)
                return
        self.build()

    def _vector_ids(self, query: str, n: int, doc_type: str | None) -> list[int]:
        kwargs = {"filter": {"type": doc_type}, "fetch_k": max(50, n * 10)} if doc_type else {}
        hits = self.store.similarity_search_with_score(query, k=n, **kwargs)
        return [doc.metadata["chunk_id"] for doc, _ in hits]

    def search(self, query: str, k: int = None, doc_type: str | None = None, mode: str | None = None):
        """Return [(Document, relevance 0-1)] best first. doc_type filters 'policy' / 'employee'."""
        self.ensure()
        k = k or settings.TOP_K
        mode = (mode or settings.RETRIEVAL_MODE).lower()
        allowed = None if doc_type is None else {
            i for i, c in enumerate(self.chunks) if c.metadata.get("type") == doc_type}
        pool = max(k * 3, 10)
        if mode == "vector":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._vector_ids(query, k, doc_type))]
        elif mode == "bm25":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self.bm25.top(query, k, allowed))]
        else:
            ranked = rrf([self._vector_ids(query, pool, doc_type), self.bm25.top(query, pool, allowed)])[:k]
        if not ranked:
            return []
        top = ranked[0][1]
        return [(self.chunks[i], round(score / top, 4)) for i, score in ranked]


rag_index = RagIndex()
