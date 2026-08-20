# Retrieval Evaluation — Analysis Log

Interpretation of each evaluation run, one entry per iteration. The raw numbers
live in `data/eval/RESULTS.md`, which is **regenerated on every run**; this file
is the hand-written "what it means" and is **never overwritten**.

Each entry is anchored to a date and a git commit so it stays reproducible, and
carries its own headline numbers so it remains self-contained even after
`RESULTS.md` is overwritten by the next run. New iterations are appended below,
each separated by a full dotted line.

## Iteration 1 — 2026-08-21 — commit `00125ae` (+ uncommitted PDF support)

**Change since baseline:** added PDF ingestion (`pypdf`) and dropped 26 academic
PDFs (~24 MB, mostly minimal-perfect-hashing papers plus a few systems/ML docs)
into the corpus. Golden set grew 51 → 76 questions; q52–q76 cover the PDFs.

**Corpus:** 3052 chunks &nbsp;|&nbsp; **Questions:** 76 &nbsp;|&nbsp; k = 5

| Config | Hit@1 | Hit@3 | MRR | nDCG@5 |
|---|---|---|---|---|
| bm25 | 0.684 | 0.789 | 0.742 | 0.763 |
| vector | 0.816 | 0.934 | 0.882 | 0.900 |
| hybrid | 0.789 | 0.921 | 0.860 | 0.881 |
| **hybrid+rerank** | **0.895** | **0.974** | **0.934** | **0.940** |

**Findings**

- **Reranking is again the highest-return component:** +8.7% MRR and +13.3% Hit@1
  over fusion, for +72 ms/query. Fusion alone is a slight net loss vs pure dense
  (−2.5% MRR) — the embedding model already handles the technical vocabulary, so
  BM25 mostly dilutes an already-correct dense ranking.
- **PDFs retrieve well.** Splitting the production path by source:
  - original `.txt` (q01–q51): Hit@1 **0.922**, MRR 0.961
  - new PDF (q52–q76): Hit@1 **0.840** (21/25), MRR 0.880
  
  The gap is corpus difficulty, not a parsing problem — extraction + chunking +
  retrieval work end-to-end on the PDFs.
- **All 4 PDF misses are sibling-paper confusion** inside the dense
  minimal-perfect-hashing cluster (~15 near-identical papers):
  - q55 (2009 *Hash, Displace, Compress*) and q66 (1996 *Family of Perfect
    Hashing*): near-misses — correct doc retrieved at **rank 2**.
  - q58 (1992 *CHM92*) and q59 (1980 *Cichelli*): **genuine misses**, not in the
    top 5 — their generic MPHF content collides with too many siblings.

**Decisions**

- Kept the misses as an honest signal of corpus density; did **not** broaden
  `relevant_docs`. Revisit only if q58/q59 specifically matter.
- Stripped the hardcoded narrative out of `RESULTS.md`; it is now tables +
  computed deltas only. Interpretation lives here from now on.

**Next candidates**

- External vector store (Qdrant) to lift the in-process / multi-worker limit.
- Consider crediting near-sibling papers in the golden set if rank-1-vs-2
  confusion proves uninteresting.

.................................................................................

<!-- Append Iteration 2 below this dotted line, same structure. -->
