# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system: ask a question, and it finds the
most relevant passages in a collection of documents and returns a grounded
answer with citations back to the sources.

Under the hood it combines **keyword search** (BM25) and **semantic search**
(vector embeddings), fuses the two rankings, re-scores them with a
**cross-encoder reranker**, and ships an **evaluation harness** that measures how
good the retrieval actually is — not just claims that it works.

**It runs with no API key.** Out of the box, answers are composed directly from
the retrieved passages (an "extractive" backend) — no credentials, no cost, no
network. Add an API key for Anthropic, OpenAI, or Google to switch on
LLM-written answers, with no code change.

---

## Quickstart

```bash
pip install -r requirements.txt
uvicorn main:app          # → http://localhost:8000  (browser UI + API)
```

The repo ships a small demo corpus, so it works immediately. Or ask a question
straight from the command line:

```bash
python -m pipeline.orchestrator --query "How does PagedAttention save memory?"
```

Or run it in Docker (model weights are baked into the image):

```bash
docker compose up --build
```

---

## How it works

There are two separate paths, because they have opposite needs — building the
index is a slow batch job, while answering a question must be fast.

```
INGEST  (build the knowledge base, run when documents change)
  documents → parse & chunk → embed → vector DB + keyword index

QUERY   (answer a question, run on every request)
  question → rewrite → hybrid search → rerank → build prompt → LLM → cite sources
```

| Stage | File | Role |
|---|---|---|
| Documents | [`documents.py`](pipeline/documents.py) | Load `.txt`/`.pdf` files, stable ids, corpus fingerprint |
| Parse & chunk | [`parse_chunk.py`](pipeline/parse_chunk.py) | Split into overlapping chunks |
| Embed | [`embed.py`](pipeline/embed.py) | MiniLM sentence embeddings |
| Vector DB | [`vector_db.py`](pipeline/vector_db.py) | FAISS similarity search, persisted to disk |
| Keyword index | [`keyword_index.py`](pipeline/keyword_index.py) | BM25 lexical search |
| Query rewrite | [`query_rewrite.py`](pipeline/query_rewrite.py) | Normalise text, expand acronyms |
| Hybrid search | [`hybrid_search.py`](pipeline/hybrid_search.py) | Fuse keyword + vector results (RRF) |
| Rerank | [`rerank.py`](pipeline/rerank.py) | Cross-encoder re-scoring of top candidates |
| Context format | [`context_format.py`](pipeline/context_format.py) | Build the numbered, source-tagged prompt |
| LLM generate | [`llm_generate.py`](pipeline/llm_generate.py) | Pluggable backends (extractive / Anthropic / OpenAI / Google) |
| Cite sources | [`cite_sources.py`](pipeline/cite_sources.py) | Map `[n]` markers back to real chunks |
| Orchestrator | [`orchestrator.py`](pipeline/orchestrator.py) | Wires it all together |

The **orchestrator** is the single entry point: `ingest()` builds the index and
`query()` answers a question.

---

## Documents & corpus data

Point the pipeline at a folder of **`.txt` or `.pdf`** files. PDF text is
extracted per page with `pypdf`. Adding another format (`.md`, `.html`, …) is one
line in the reader registry in `documents.py` — nothing else changes.

This repo tracks two things:

- A small **`.txt` demo corpus** in `data/sample_docs/`, so the pipeline, tests,
  and CI all run on a fresh clone.
- The labelled **golden set** in `data/eval/golden_set.json`, which lists exactly
  which document answers each evaluation question.

The **full evaluation corpus** (26 academic PDFs) is large and third-party, so it
is *not* stored in git. Download it and drop the files into `data/sample_docs/`:

> **Corpus download:** https://drive.google.com/drive/folders/19ctQULoc0GW9M9MpEqVjeMwQiZ_hiHjs

(CI fetches this automatically before running the evaluation.)

---

## Results

The point of the evaluation harness is to check whether each part earns its
cost. Latest run — four retrieval strategies over **76 labelled questions**
(3,052 chunks). Reproduce with `make eval` (no API key needed, fully
deterministic).

| Configuration | Hit@1 | Hit@3 | MRR | Avg latency |
|---|---|---|---|---|
| BM25 only (keyword) | 0.684 | 0.789 | 0.742 | 4 ms |
| Vector only (semantic) | 0.816 | 0.934 | 0.882 | 10 ms |
| Hybrid (keyword + vector) | 0.789 | 0.921 | 0.860 | 12 ms |
| **Hybrid + rerank** | **0.895** | **0.974** | **0.934** | 84 ms |

- **Hit@1** = how often the single best result is a correct document.
- **MRR** = rewards ranking the right document higher.

**Reranking is the biggest win** (+13.3% Hit@1 over hybrid alone). The raw
tables are auto-generated in [`data/eval/RESULTS.md`](data/eval/RESULTS.md); the
written analysis of each run is kept in
[`evaluation/ANALYSIS.md`](evaluation/ANALYSIS.md).

---

## Enabling a real LLM

By default, answers are extractive (offline, free). To use a hosted or local
model instead, copy the example env file and fill in **one** provider block:

```bash
cp .env.example .env
```

```bash
# in .env — pick ONE provider and paste your key
RAG_LLM_PROVIDER=anthropic
RAG_LLM_API_KEY=sk-ant-...
RAG_LLM_MODEL=claude-sonnet-5
```

Other providers use the same three variables:

| Provider | `RAG_LLM_PROVIDER` | Extra | Example model |
|---|---|---|---|
| Anthropic | `anthropic` | — | `claude-sonnet-5` |
| OpenAI | `openai` | — | `gpt-4o-mini` |
| Google Gemini | `google` | — | `gemini-2.5-flash` |
| Local (vLLM, Ollama, …) | `openai` | `RAG_LLM_BASE_URL=http://localhost:11434/v1` | `llama3.1` |

The active backend is reported on every response, so an extractive answer is
never mistaken for a generated one. Any setting in [`config.py`](config.py) can
be overridden with a `RAG_` prefix.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser UI |
| `POST /query` | Answer a question — returns answer, citations, timings, backend |
| `POST /ingest` | Index a directory of documents |
| `GET /health` | Index size and active generation backend |
| `GET /docs` | Interactive API documentation |

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid search?", "mode": "hybrid", "rerank": true}'
```

Index your own documents:

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"directory": "./my-docs"}'
```

---

## Testing & evaluation

```bash
make test               # unit + integration tests
make eval               # retrieval metrics against the golden set
pytest -m "not slow"    # fast unit tests only
```

CI runs the tests **and** the evaluation on every push, so a change that quietly
hurts retrieval quality shows up as a failing build.

---

## Limitations

- **In-process indexes** — with multiple workers, each holds its own copy in
  memory. This is what motivates moving to an external vector store (Qdrant is
  the intended next step).
- **BM25 rebuilds fully on every ingest** — fine for batch loading, wrong for a
  high-write workload.
- **PDF extraction is text-layer only** — scanned or image-only PDFs yield no
  text and are skipped; there is no OCR and no table/layout awareness.
- **No generation-quality evaluation** — judging answer quality needs an LLM
  judge (costs money, breaks the no-key guarantee). Retrieval quality caps
  everything downstream, so it came first.

---

## License

MIT — see [LICENSE](LICENSE).
