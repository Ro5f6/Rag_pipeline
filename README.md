# RAG Pipeline

A production-shaped Retrieval-Augmented Generation system — hybrid retrieval,
reciprocal rank fusion, cross-encoder reranking, and grounded answers with
auditable citations — plus an evaluation harness that measures whether the
retrieval actually works.

**Runs with no API key.** Generation falls back to an extractive backend, so the
full retrieval stack works offline, at no cost. Add a key to switch on LLM
generation — no code change.

## Quickstart

```bash
pip install -r requirements.txt
uvicorn main:app          # → http://localhost:8000  (browser UI + API)
```

Or from the command line:

```bash
python -m pipeline.orchestrator --query "How does PagedAttention save memory?"
```

Or in Docker (model weights baked into the image):

```bash
docker compose up --build
```

### Corpus data

The evaluation corpus (`data/sample_docs/`) is **not** tracked in git — it is
large and partly third-party. Download it and drop the files into
`data/sample_docs/` before running ingest or `make eval`:

> **Corpus:** https://drive.google.com/drive/folders/19ctQULoc0GW9M9MpEqVjeMwQiZ_hiHjs

The labelled golden set in `data/eval/golden_set.json` stays in the repo and
documents exactly which files the evaluation expects.

## Results

Four retrieval configurations over a 20-document corpus, 51 labelled questions.
Reproduce with `make eval` — deterministic, no API key.

| Configuration | Hit@1 | Hit@3 | MRR | Avg latency |
|---|---|---|---|---|
| BM25 only | 0.667 | 0.941 | 0.794 | <1 ms |
| Vector only | 0.902 | 0.980 | 0.946 | 11 ms |
| Hybrid (RRF) | 0.902 | 1.000 | 0.944 | 8 ms |
| **Hybrid + rerank** | **0.941** | **1.000** | **0.971** | 72 ms |

**Reranking is the highest-return component** (+4.3% Hit@1 over fusion, for
~64 ms). Full per-query-type analysis lives in [`evaluation/`](evaluation/).

## Architecture

One file per stage, one responsibility each. Ingest and query are separate paths
because they have opposite profiles — batch vs. latency-critical.

```
INGEST   documents → parse_chunk → embed → vector_db + keyword_index

QUERY    query_rewrite → hybrid_search → rerank → context_format
                              ▲                        │
                    vector_db ┘ keyword_index          ▼
                                        llm_generate → cite_sources
```

| Stage | File | Role |
|---|---|---|
| Documents | [`documents.py`](pipeline/documents.py) | Load files, stable ids, corpus fingerprint |
| Parse & chunk | [`parse_chunk.py`](pipeline/parse_chunk.py) | Recursive splitting with overlap |
| Embed | [`embed.py`](pipeline/embed.py) | MiniLM embeddings, L2-normalised |
| Vector DB | [`vector_db.py`](pipeline/vector_db.py) | FAISS `IndexFlatIP`, persisted |
| Keyword index | [`keyword_index.py`](pipeline/keyword_index.py) | BM25 lexical retrieval |
| Query rewrite | [`query_rewrite.py`](pipeline/query_rewrite.py) | Normalise, expand acronyms |
| Hybrid search | [`hybrid_search.py`](pipeline/hybrid_search.py) | Reciprocal rank fusion |
| Rerank | [`rerank.py`](pipeline/rerank.py) | Cross-encoder re-scoring |
| Context format | [`context_format.py`](pipeline/context_format.py) | Numbered, source-tagged prompt |
| LLM generate | [`llm_generate.py`](pipeline/llm_generate.py) | Provider-agnostic backends |
| Cite sources | [`cite_sources.py`](pipeline/cite_sources.py) | Map `[n]` markers to chunks |
| Orchestrator | [`orchestrator.py`](pipeline/orchestrator.py) | Wires the diagram together |

The orchestrator is the single entry point — `ingest()` builds the knowledge
base, `query()` answers a question. Persisted indexes carry a manifest that
triggers an automatic rebuild when the corpus, embedding model, or chunk
settings change.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser UI |
| `POST /query` | Answer a question — returns answer, citations, timings, backend |
| `POST /ingest` | Index a directory of documents |
| `GET /health` | Index size and active generation backend |
| `GET /docs` | OpenAPI documentation |

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid search?", "mode": "hybrid", "rerank": true}'
```

Use your own documents:

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"directory": "./my-docs"}'
```

## Enabling a real LLM

Set one of these — nothing else changes. The active backend is reported on every
response, so an extractive answer is never mistaken for a generated one.

```bash
export ANTHROPIC_API_KEY=sk-ant-...                    # Anthropic

export OPENAI_API_KEY=sk-... RAG_LLM_PROVIDER=openai   # OpenAI

export RAG_LLM_PROVIDER=openai \                        # local (vLLM, Ollama, …)
       RAG_LLM_BASE_URL=http://localhost:11434/v1 \
       RAG_LLM_MODEL=llama3.1
```

Any field in [`config.py`](config.py) is overridable with a `RAG_` prefix — see
[`.env.example`](.env.example).

## Testing & evaluation

```bash
make test     # unit + integration tests
make eval     # retrieval metrics against the golden set
pytest -m "not slow"    # fast unit tests only
```

CI runs both on every push, so retrieval quality is a regression gate.

## Limitations

- **In-process indexes** — under multiple workers each holds its own copy; the
  motivation for an external vector store (Qdrant is the intended next step).
- **Small corpus** (20 docs) — Hit@5 saturates, so Hit@1 is the discriminating
  metric.
- **BM25 rebuilds fully on every ingest** — fine for batch, wrong for high-write.
- **`.txt` and `.pdf`** — PDF text is extracted per-page via `pypdf`; scanned
  or image-only PDFs (no text layer) are skipped, and no layout/table awareness.
- **No generation-quality eval** — would need an LLM judge, breaking the no-key
  guarantee; retrieval metrics cap everything downstream, so they came first.

## License

MIT — see [LICENSE](LICENSE).
