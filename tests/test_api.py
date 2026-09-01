"""
tests/test_api.py
==================
HTTP contract tests.

The pipeline is replaced with a stub so these test the API layer only -- status
codes, validation, and serialization -- without loading transformer models.
Fast tests get run; slow ones get skipped, and a test suite nobody runs
protects nothing.
"""

import pytest
from fastapi.testclient import TestClient

import main
from pipeline.cite_sources import Citation
from pipeline.orchestrator import RAGResponse


class _StubVectorDB:
    def __init__(self, size: int = 3):
        self.size = size

    def __len__(self):
        return self.size


class _StubGenerator:
    name = "groq"
    model = "llama-3.3-70b-versatile"


class _StubPipeline:
    """Stands in for RAGPipeline with the same surface the API depends on."""

    def __init__(self, chunks: int = 3):
        self.vector_db = _StubVectorDB(chunks)
        self.generator = _StubGenerator()
        self.last_call = {}

    def load_or_ingest(self, directory=None):
        return len(self.vector_db)

    def ingest(self, directory, persist=None):
        if directory == "/does/not/exist":
            raise FileNotFoundError(f"No documents found in {directory!r}")
        return 7

    def query(self, user_query, mode="hybrid", use_rerank=True):
        self.last_call = {"query": user_query, "mode": mode, "rerank": use_rerank}
        return RAGResponse(
            answer="An answer grounded in the context [1].",
            citations=[Citation(marker="[1]", filename="a.txt", chunk_id="c1", text_preview="preview")],
            backend="groq",
            model="llama-3.3-70b-versatile",
            retrieval_mode=mode,
            rewritten_query=user_query.lower(),
            candidates_retrieved=20,
            chunks_used=5,
            timings_ms={"retrieval": 12.0, "generation": 0.3, "total": 12.3},
        )


@pytest.fixture
def client(monkeypatch):
    stub = _StubPipeline()
    monkeypatch.setattr(main, "pipeline", stub)
    with TestClient(main.app) as c:
        c.stub = stub
        yield c


# --------------------------------------------------------------------- #
# Health & config
# --------------------------------------------------------------------- #
def test_health_reports_index_size_and_active_backend(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 3
    assert body["generation_backend"] == "groq"


def test_config_endpoint_describes_the_running_instance(client):
    body = client.get("/config").json()
    assert "embedding_model" in body and "rrf_k" in body


# --------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------- #
def test_query_returns_answer_citations_and_per_stage_timings(client):
    body = client.post("/query", json={"query": "What is RAG?"}).json()

    assert body["answer"]
    assert body["citations"][0]["marker"] == "[1]"
    assert body["backend"] == "groq"
    assert "total" in body["timings_ms"]
    assert body["candidates_retrieved"] == 20


def test_query_forwards_retrieval_options_to_the_pipeline(client):
    client.post("/query", json={"query": "x", "mode": "vector", "rerank": False})

    assert client.stub.last_call == {"query": "x", "mode": "vector", "rerank": False}


def test_query_defaults_to_hybrid_with_reranking(client):
    client.post("/query", json={"query": "x"})
    assert client.stub.last_call["mode"] == "hybrid"
    assert client.stub.last_call["rerank"] is True


def test_unknown_retrieval_mode_is_rejected(client):
    assert client.post("/query", json={"query": "x", "mode": "telepathy"}).status_code == 422


def test_empty_query_is_rejected(client):
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_query_against_an_empty_index_returns_409(monkeypatch):
    monkeypatch.setattr(main, "pipeline", _StubPipeline(chunks=0))
    with TestClient(main.app) as c:
        response = c.post("/query", json={"query": "anything"})

    assert response.status_code == 409
    assert "ingest" in response.json()["detail"].lower()


def test_generation_failure_surfaces_as_502(client, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(client.stub, "query", explode)

    response = client.post("/query", json={"query": "x"})
    assert response.status_code == 502
    assert "provider unavailable" in response.json()["detail"]


# --------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------- #
def test_ingest_reports_chunk_counts(client):
    body = client.post("/ingest", json={"directory": "./data/sample_docs"}).json()
    assert body["chunks_ingested"] == 7


def test_ingesting_a_missing_directory_returns_400(client):
    assert client.post("/ingest", json={"directory": "/does/not/exist"}).status_code == 400


# --------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------- #
def test_root_serves_the_demo_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
