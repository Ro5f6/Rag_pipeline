"""
main.py
========
HTTP surface for the pipeline: a FastAPI service plus a small browser UI.

Kept deliberately thin. Its only job is to parse HTTP in, call the pipeline,
and serialize HTTP out -- all real logic lives in pipeline/orchestrator.py.
That separation is what allows the same pipeline to be driven by the CLI, the
evaluation harness, and this API without any of them duplicating each other.

Run:
    uvicorn main:app --reload
    open http://localhost:8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import settings
from pipeline.observability import configure_logging
from pipeline.orchestrator import RAGPipeline

logger = logging.getLogger(__name__)

# One shared pipeline per process. Models take seconds to load and hundreds of
# megabytes to hold, so they are loaded once at startup rather than per
# request. Note the consequence of an in-process index: under `gunicorn -w 4`
# each worker holds its own copy, so a document ingested through one worker is
# invisible to the other three. That is the exact constraint that motivates an
# external vector store, and it is called out in the README rather than left
# as a surprise.
pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the knowledge base before the first request, not during it."""
    configure_logging(settings.log_level)
    try:
        chunks = pipeline.load_or_ingest()
        logger.info("Startup complete: %d chunks available.", chunks)
    except FileNotFoundError:
        # An empty corpus is a valid state -- the service starts and /ingest
        # can be called later. Failing startup here would be worse.
        logger.warning("No corpus found at %s. Call /ingest to populate.", settings.auto_ingest_dir)
    yield


app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    description="Hybrid retrieval with reciprocal rank fusion, cross-encoder reranking, and grounded citations.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    directory: str = Field(default="./data/sample_docs", description="Directory of .txt or .pdf documents")


class IngestResponse(BaseModel):
    chunks_ingested: int
    total_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The user's question")
    mode: str = Field(default="hybrid", description="'hybrid', 'vector', or 'keyword'")
    rerank: bool = Field(default=True, description="Apply cross-encoder reranking")


class CitationResponse(BaseModel):
    marker: str
    filename: str
    chunk_id: str
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    backend: str                        # which generator produced the answer
    model: str
    retrieval_mode: str
    rewritten_query: str
    candidates_retrieved: int
    chunks_used: int
    timings_ms: Dict[str, float]        # per-stage latency, not just a total


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    generation_backend: str
    model: str


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #
@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        n = pipeline.ingest(req.directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(chunks_ingested=n, total_chunks=len(pipeline.vector_db))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if len(pipeline.vector_db) == 0:
        raise HTTPException(status_code=409, detail="No documents indexed yet. Call /ingest first.")

    if req.mode not in {"hybrid", "vector", "keyword"}:
        raise HTTPException(status_code=422, detail=f"Unknown mode: {req.mode!r}")

    try:
        result = pipeline.query(req.query, mode=req.mode, use_rerank=req.rerank)
    except Exception as exc:  # noqa: BLE001
        # A provider outage or a bad API key should surface as a clean 502
        # rather than an opaque stack trace.
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                marker=c.marker,
                filename=c.filename,
                chunk_id=c.chunk_id,
                text_preview=c.text_preview,
            )
            for c in result.citations
        ],
        backend=result.backend,
        model=result.model,
        retrieval_mode=result.retrieval_mode,
        rewritten_query=result.rewritten_query,
        candidates_retrieved=result.candidates_retrieved,
        chunks_used=result.chunks_used,
        timings_ms=result.timings_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_indexed=len(pipeline.vector_db),
        generation_backend=pipeline.generator.name,
        model=pipeline.generator.model,
    )


@app.get("/config")
def show_config() -> dict:
    """Effective retrieval settings, so a running instance is self-describing."""
    return {
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "hybrid_top_k": settings.hybrid_top_k,
        "rrf_k": settings.rrf_k,
        "rerank_top_k": settings.rerank_top_k,
        "llm_provider": settings.llm_provider,
    }


_UI_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    if not _UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(_UI_FILE)
