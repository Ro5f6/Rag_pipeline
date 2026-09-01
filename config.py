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
    embedding_device: str = Field(
        default="auto",
        description="Torch device for the encoder: auto | cpu | cuda | mps (Apple Silicon GPU)",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Chunks per encode batch; larger batches use the GPU more fully",
    )

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
    # Generation goes through LangChain's init_chat_model, so any provider is a
    # config change, not a code change. There is no offline fallback: a model
    # must be configured for the /query path (retrieval and eval run without one).
    # See pipeline/llm_generate.py.
    llm_provider: str = Field(
        default="groq",
        description="LangChain provider id: groq | ollama | openai | google_genai | anthropic. "
                    "OpenAI-compatible servers (vLLM, Together, Fireworks, OpenRouter) use 'openai' + a base_url. "
                    "'auto' infers from the model name.",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model id, e.g. 'llama-3.3-70b-versatile' (groq), 'llama3.1' (ollama), 'gemini-3.6-flash' (google).",
    )
    llm_base_url: str = Field(
        default="",
        description="Custom endpoint for OpenAI-compatible servers, e.g. http://localhost:8000/v1 (vLLM) or :11434/v1 (Ollama).",
    )
    llm_api_key: str = Field(
        default="",
        description="Optional. Falls back to the provider's standard env var (GROQ_API_KEY, OPENAI_API_KEY, ...).",
    )
    llm_temperature: float = Field(default=0.0, description="Sampling temperature; 0.0 for the most grounded answers.")
    llm_max_tokens: int = Field(default=1024)

    # ---- Generation evaluation (RAGAS) ----
    # The judge that SCORES answers -- deliberately a different model/family from
    # the generator under test, to avoid self-preference bias. Used only by
    # evaluation/eval_generation.py, never on the serving path.
    judge_provider: str = Field(default="groq", description="LangChain provider id for the RAGAS judge model.")
    judge_model: str = Field(default="openai/gpt-oss-120b", description="Judge model id.")
    judge_api_key: str = Field(default="", description="Judge key; falls back to the provider's env var (e.g. GOOGLE_API_KEY).")
    judge_base_url: str = Field(default="", description="Custom judge endpoint, if any.")
    judge_temperature: float = Field(default=0.0, description="Judge temperature; 0.0 to keep scoring stable across runs.")
    judge_max_tokens: int = Field(default=4096, description="Max tokens for the judge; reasoning-model judges need headroom to finish the JSON verdict.")
    judge_structured_mode: str = Field(
        default="md_json",
        description="How the judge is asked for structured output: md_json (universal, works with LM Studio/local) | "
                    "json | json_schema | tools (native modes some hosted providers support).",
    )
    eval_sample_size: int = Field(default=0, description="Questions to evaluate: 0 = all, else the first N (rate-limit control).")
    eval_max_workers: int = Field(default=4, description="RAGAS concurrency; keep low for rate-limited judge endpoints.")

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
