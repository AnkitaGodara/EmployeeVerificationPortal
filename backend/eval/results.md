# RAG retrieval evaluation (2026-11-21)

28 labelled questions · 43 chunks · embeddings: `tfidf` · k = 4

| Retriever | Hit@1 | Hit@4 | MRR |
|---|---|---|---|
| vector | 57% | 75% | 0.64 |
| bm25 | 57% | 79% | 0.66 |
| hybrid | 57% | 82% | 0.67 |

**vector missed:** Which areas can an L3 employee get into?; Can I work in the office on a Sunday night?; How long does it take to get L4 approved?; How long are verification records stored?; Can I refuse to give my face photo?; A former employee shows up at reception. How do we treat them?; What number do I call in an emergency?

**bm25 missed:** Which areas can an L3 employee get into?; Someone is suspended - can they come into the building?; Can I work in the office on a Sunday night?; How long does it take to get L4 approved?; How long are verification records stored?; Can I refuse to give my face photo?

**hybrid missed:** Which areas can an L3 employee get into?; Can I work in the office on a Sunday night?; How long does it take to get L4 approved?; How long are verification records stored?; Can I refuse to give my face photo?
