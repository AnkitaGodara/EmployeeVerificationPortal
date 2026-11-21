"""BM25 keyword search (Okapi BM25) and Reciprocal Rank Fusion.

Vector search is good at meaning ("not permitted" ~ "denied"); BM25 is good at exact words
(codes, "L4", "Unknown"). Hybrid = run both and fuse the rankings with RRF:
    score(doc) = sum over rankers of 1 / (60 + rank)
"""
import math
import re
from collections import Counter

STOP = set("""a an the and or of to in on for is are was were be been with by at as it its this that
these those from can may must not no do does did if then than so such any all each per what which who
how when where why should would could will shall their they them he she his her you your we our""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(t) for t in texts]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(term for doc in self.docs for term in set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(doc) for doc in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s, dl = 0.0, len(doc)
            for term in q:
                f = tf.get(term, 0)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top(self, query: str, k: int, allowed: set[int] | None = None) -> list[int]:
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0 and (allowed is None or i in allowed)]
        return [i for _, i in sorted(scored, reverse=True)[:k]]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda t: -t[1])
