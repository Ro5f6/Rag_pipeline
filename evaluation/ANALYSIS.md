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

## Iteration 2 — 2026-08-24 — reranker bake-off (no config change kept)

Goal: raise Hit@1 (0.895). First **diagnosed where the misses come from**, then
tested reranker swaps. All runs are throwaway (env-var overrides, baseline
`RESULTS.md` restored after each).

**Diagnosis (new tool: `evaluation/diagnose_retrieval.py`)**

- **Pool recall@20 = 0.974** vs **Hit@1 = 0.895** → for 74/76 questions the answer
  is already in the candidate pool. Retrieval breadth is *not* the bottleneck for
  the 6 reranker misses.
- The 8 misses split into two causes:
  - **6 = reranker** (doc in pool at rank 1–4, demoted to #2). A bigger pool
    cannot help these.
  - **2 = pool too shallow** (q58, q59). Their documents rank **12–16 of 46**
    (vector 13/12, BM25 16/8), i.e. retrievable and *not* buried — but their best
    chunk sits just past the 20-**chunk** pool (top docs contribute many chunks,
    so 20 chunks ≈ ~13 distinct docs). Fix = a modest `hybrid_top_k` bump, **not**
    a new embedder.

**Reranker bake-off (hybrid+rerank, 76 questions)**

| Reranker | Hit@1 | Hit@3 | MRR | nDCG@5 | Latency |
|---|---|---|---|---|---|
| **ms-marco-MiniLM-L-6 (baseline)** | **0.895** | **0.974** | **0.934** | **0.940** | **84 ms** |
| bge-reranker-base | 0.868 | 0.934 | 0.908 | 0.924 | 372 ms |
| bge-reranker-large | 0.868 | 0.974 | 0.921 | 0.932 | 1094 ms |

**Finding:** the baseline wins **every** quality column *and* is fastest. Both
BGE models lost; bge-base even drops Hit@3 (pushes correct docs below rank 3).
Reason: `ms-marco-MiniLM` is trained on exactly this task (English Q&A passage
ranking), while BGE is general/multilingual. **Task-match beat model size** — the
same lesson as "hybrid fusion ≈ wash" from iteration 1.

**Decisions**

- **Keep the baseline reranker.** Reject BGE (worse + slower).
- One reranker try left in the same family (`ms-marco-MiniLM-L-12`) — deferred.
- Move the effort to the **embedding model** (for semantic/mixed misses) and the
  **`hybrid_top_k` bump** (proven fix for q58/q59).

.................................................................................

## Iteration 3 — 2026-08-25 — commit `1c8713f` (+ uncommitted scale-up)

**Change since Iteration 1:** scaled the corpus ~20× — added ~184 academic PDFs
(systems, ML, data-structures papers) on top of the original set. Golden set grew
76 → 135 questions (dropped q133, which targeted a non-extractable scanned PDF).
7 PDFs are non-extractable (4 scanned/image-only, 3 pypdf failures) and are
skipped at load time. Ingestion now runs the encoder on the **Apple M2 GPU (MPS)**
— ~3.5× faster embedding than CPU on this hardware — and the index is **persisted
once** (`load_or_ingest`) so eval runs reuse it instead of re-embedding.

**Corpus:** 63,378 chunks (223 docs) &nbsp;|&nbsp; **Questions:** 135 &nbsp;|&nbsp; k = 5

| Config | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | Latency |
|---|---|---|---|---|---|---|
| bm25 | 0.696 | 0.830 | 0.837 | 0.756 | 0.776 | 162 ms |
| vector | 0.778 | 0.911 | 0.941 | 0.849 | 0.869 | 15 ms |
| hybrid | 0.763 | 0.933 | 0.941 | 0.842 | 0.867 | 152 ms |
| **hybrid+rerank** | **0.904** | **0.985** | **0.985** | **0.943** | **0.951** | 218 ms |

**Findings**

- **The metrics stabilized under scale — the headline result.** Against
  Iteration 1 (76 questions / 3,052 chunks), a ~20× larger index and nearly 2×
  the questions did **not** degrade retrieval; hybrid+rerank held and slightly
  improved: Hit@1 0.895 → **0.904**, Hit@3 0.974 → **0.985**, MRR 0.934 →
  **0.943**, nDCG@5 0.940 → **0.951**. The pipeline scales without falling apart.
- **Both prior lessons reconfirmed at scale:**
  - Reranking is the highest-return component — **+18.4% Hit@1** and **+12.0%
    MRR** over fusion for +66 ms/query.
  - Fusion is still a slight net loss vs pure dense (**−0.8% MRR**); BM25 keeps
    diluting an already-strong dense ranking on this technical vocabulary.
- **Reranking's real job is rescuing semantic (paraphrase) queries.** By query
  type, MRR: lexical is near-solved (bm25 0.808 → rerank **0.981**); semantic is
  the weakest class and gains the most from reranking (bm25 0.531 → hybrid 0.703
  → rerank **0.919**). Future quality headroom lives in the embedder for the
  semantic/mixed classes, consistent with the Iteration 2 diagnosis.
- **Latency grew where expected.** hybrid+rerank rose 84 ms → 218 ms vs the
  small-corpus runs — driven by BM25 scanning 63k chunks (162 ms); dense search
  stayed 15 ms thanks to the flat FAISS index. Still comfortably real-time.

**Decisions**

- **Keep the persisted-index eval path.** One GPU ingest (~3–4 min) now backs
  every eval run; the manifest auto-rebuilds only on corpus/model/chunk changes.
- Kept the 135-question set as-is; did not broaden `relevant_docs` for the
  remaining sibling-paper misses.

**Next candidates**

- Parallelize / swap PDF extraction (pypdf → PyMuPDF or multiprocessing): at 63k
  chunks the ~150 s CPU extraction, not GPU embedding, is now the ingest floor.
- Embedding-model swap for the semantic class (gte-base was a better retriever in
  earlier trials; revisit now that fusion behaviour is understood).
- Generation-quality eval (the retrieval ceiling is high; measure the answer
  stage next).

.................................................................................

<!-- Append Iteration 4 below this dotted line, same structure. -->
