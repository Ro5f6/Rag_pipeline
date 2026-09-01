# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system: ask a question, and it finds the
most relevant passages in a collection of documents and returns a grounded
answer with citations back to the sources.

Under the hood it combines **keyword search** (BM25) and **semantic search**
(vector embeddings), fuses the two rankings, re-scores them with a
**cross-encoder reranker**, and ships an **evaluation harness** that measures how
good the retrieval actually is — not just claims that it works.

**Generation is model-agnostic.** Answers are written by an LLM chosen through
LangChain's `init_chat_model`, so switching between open-source models (Groq,
Ollama, any OpenAI-compatible server) and hosted ones (Gemini, Claude) is a
config change, not a code change. Retrieval and the evaluation harness run with
no key at all.

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
| LLM generate | [`llm_generate.py`](pipeline/llm_generate.py) | Any model via LangChain `init_chat_model` (Groq / Ollama / OpenAI-compatible / Gemini / Claude) |
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

| Configuration | Hit@1 | Hit@3 | MRR | nDCG@5 | Avg latency |
|---|---|---|---|---|---|
| BM25 only (keyword) | 0.684 | 0.789 | 0.742 | 0.763 | 4 ms |
| Vector only (semantic) | 0.816 | 0.934 | 0.882 | 0.900 | 10 ms |
| Hybrid (keyword + vector) | 0.789 | 0.921 | 0.860 | 0.881 | 12 ms |
| **Hybrid + rerank** | **0.895** | **0.974** | **0.934** | **0.940** | 84 ms |

- **Hit@1** = how often the single best result is a correct document.
- **MRR** = rewards ranking the right document higher.
- **nDCG@5** = rewards packing the relevant documents higher within the top 5.

**Reranking is the biggest win** (+13.3% Hit@1 over hybrid alone). The raw
tables are auto-generated in [`data/eval/RESULTS.md`](data/eval/RESULTS.md); the
written analysis of each run is kept in
[`evaluation/ANALYSIS.md`](evaluation/ANALYSIS.md).

---

## Choosing a model

Generation goes through LangChain's `init_chat_model`, so any provider is just
two variables. Copy the example env file and set **one** provider block:

```bash
cp .env.example .env
```

```bash
# in .env — open-source model on Groq's hosted endpoint
RAG_LLM_PROVIDER=groq
RAG_LLM_API_KEY=gsk_...
RAG_LLM_MODEL=llama-3.3-70b-versatile
```

Every provider uses the same variables:

| Provider | `RAG_LLM_PROVIDER` | Extra | Example model |
|---|---|---|---|
| Groq (hosted OSS) | `groq` | — | `llama-3.3-70b-versatile` |
| Ollama (local OSS) | `ollama` | — | `llama3.1` |
| OpenAI-compatible (vLLM, Together, Fireworks, OpenRouter) | `openai` | `RAG_LLM_BASE_URL=…` | `meta-llama/Llama-3.1-8B-Instruct` |
| Google Gemini | `google_genai` | — | `gemini-2.5-flash` |
| Anthropic | `anthropic` | — | `claude-sonnet-5` |

A model must be configured for `/query`; retrieval and `make eval` need no key.
The active provider and model are reported on every response, and any setting in
[`config.py`](config.py) can be overridden with a `RAG_` prefix.

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

## Next steps

Where the project is today, and what we're planning to work on next:

- **External vector store (Qdrant).** Indexes are currently held in memory, so
  under multiple workers each keeps its own copy. Moving to a shared external
  store lets the pipeline scale horizontally — this is the next priority.
- **Incremental keyword indexing.** BM25 currently rebuilds fully on every
  ingest, which is fine for batch loading but wrong for a high-write workload;
  we plan to move to an engine that indexes incrementally (e.g. OpenSearch).
- **OCR and layout-aware PDF parsing.** Extraction is text-layer only today, so
  scanned/image PDFs and tables aren't handled. Adding OCR and layout awareness
  would open up a much wider range of real-world documents.
- **Generation-quality evaluation.** The harness measures retrieval only. Adding
  an opt-in LLM-judge for faithfulness and answer relevance would evaluate the
  generated answers too, while keeping the default no-key path free.

---

## License

MIT — see [LICENSE](LICENSE).
