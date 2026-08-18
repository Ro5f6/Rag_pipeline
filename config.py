"""
config.py
=========
Central configuration for the whole pipeline.

Why this exists (an operational habit, not just style):
magic numbers -- chunk size, top_k, model names -- scattered across modules are
impossible to tune safely, because changing retrieval behaviour means editing
code and redeploying. One Settings object, loaded once, makes every stage
independently tunable and overridable from the environment (12-factor style),
so the same image can run with different retrieval parameters per environment:

    RAG_RERANK_TOP_K=8 RAG_CHUNK_SIZE=800 uvicorn main:app

Every field below can be set as an environment variable with the RAG_ prefix,
or written into a .env file (see .env.example).
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- Parse & chunk ----
    # Splitting is always recursive (paragraph -> line -> sentence -> space),
    # with no alternative strategy. See pipeline/parse_chunk.py for why a
    # silent fallback would undermine the retrieval metrics.
    chunk_size: int = Field(default=500, description="Target characters per chunk")
    chunk_overlap: int = Field(default=50, description="Overlap between consecutive chunks")

    # ---- Embed ----
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # ---- Vector DB ----
    # vector_dim is deliberately NOT hardcoded into the pipeline: the index is
    # built lazily from the embedder's actual output dimension, so swapping
    # embedding_model can never silently desync the index shape.
    vector_dim: int = Field(default=384, description="Fallback dim; the embedder's real dim wins")

    # ---- Persistence ----
    index_dir: str = Field(
        default="./data/index",
        description="Directory holding the persisted FAISS index and BM25 corpus",
    )
    persist_indexes: bool = Field(
        default=True,
        description="Save indexes after ingest and reload them on startup",
    )

    # ---- Hybrid search ----
    hybrid_top_k: int = Field(default=20, description="Candidates pulled from EACH retriever before fusion")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant (standard default = 60)")

    # ---- Rerank ----
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_top_k: int = Field(default=5, description="Final number of chunks passed to the LLM")

    # ---- LLM generate ----
    # The generation provider is a swappable detail. The pipeline runs fully
    # without any credentials: generation falls back to an extractive answer
    # composed from the retrieved chunks. Drop in a key (or point llm_base_url
    # at a self-hosted server) to switch on real model generation -- no code
    # change required. See pipeline/llm_generate.py for the backend registry.
    llm_provider: str = Field(
        default="auto",
        description="'auto', 'anthropic', 'openai' (any OpenAI-compatible server), or 'extractive'",
    )
    llm_model: str = Field(
        default="",
        description="Model id. Blank uses the provider's default (e.g. claude-sonnet-5, gpt-4o-mini).",
    )
    llm_base_url: str = Field(
        default="",
        description="Custom endpoint, e.g. http://localhost:8000/v1 for vLLM or :11434/v1 for Ollama",
    )
    llm_api_key: str = Field(
        default="",
        description="Optional. Falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY in the environment.",
    )
    llm_max_tokens: int = Field(default=1024)
    llm_timeout_seconds: float = Field(default=60.0)

    # ---- Service ----
    log_level: str = Field(default="INFO")
    auto_ingest_dir: str = Field(
        default="./data/sample_docs",
        description="Corpus ingested on first startup when no persisted index exists",
    )

    model_config = {
        "env_prefix": "RAG_",
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
