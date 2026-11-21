"""Embedding models. Pick one with EMBEDDINGS_PROVIDER:

* tfidf (default): scikit-learn TF-IDF, fitted on our own documents. No download, no API key,
  tiny memory - runs on free hosting. Good for keyword-heavy policy text.
* fastembed: BAAI/bge-small-en-v1.5 via ONNX (semantic, ~70 MB download on first run).
* huggingface: sentence-transformers all-MiniLM-L6-v2 (semantic, needs PyTorch).
"""
from pathlib import Path

import joblib
import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings(Embeddings):
    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]) -> "TfidfEmbeddings":
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words="english", min_df=1)
        self.vectorizer.fit(texts)
        return self

    def _embed(self, texts: list[str]) -> list[list[float]]:
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).tolist()         # unit vectors: L2 distance ranks like cosine

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def save(self, folder: Path):
        joblib.dump(self.vectorizer, folder / "tfidf.joblib")

    @classmethod
    def load(cls, folder: Path) -> "TfidfEmbeddings":
        return cls(joblib.load(folder / "tfidf.joblib"))


def make_embeddings(provider: str, corpus: list[str] | None = None, folder: Path | None = None):
    """Create (when corpus is given) or load (when only folder is given) the embedding model."""
    provider = provider.lower()
    if provider == "tfidf":
        if corpus is not None:
            emb = TfidfEmbeddings().fit(corpus)
            if folder:
                emb.save(folder)
            return emb
        return TfidfEmbeddings.load(folder)
    if provider == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     encode_kwargs={"normalize_embeddings": True})
    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider}")
