"""
evaluation/ragas_judge.py
=========================
Builds the two things RAGAS needs to score answers: a judge LLM and an
embeddings model, using RAGAS 0.4.x's modern (non-deprecated) provider API.

  - Judge LLM: built with `llm_factory`, which returns an instructor-backed LLM
    that forces structured (JSON) output. Every provider is routed through its
    OpenAI-compatible endpoint (base_url below) so we use one reliable code path
    -- instructor's JSON mode -- and sidestep RAGAS's native groq/gemini adapters
    (which are broken against the installed instructor) and the litellm dependency.
  - Embeddings: RAGAS-native HuggingFaceEmbeddings, loading the project's local
    MiniLM model. Reused so answer-relevancy costs no API calls and matches
    retrieval.

Kept separate from the generator (RAG_JUDGE_* vs RAG_LLM_*): scoring a model's
answers with a different model avoids self-preference bias.

`import evaluation._ragas_compat` must run before `import ragas`; it patches a
dead Vertex import RAGAS 0.4.x still requires. See that module for why.
"""

from __future__ import annotations

from typing import Optional, Tuple

from config import Settings
from pipeline.embed import resolve_device

import instructor

import evaluation._ragas_compat  # noqa: F401  (installs the Vertex shim on import)
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM, InstructorLLM, InstructorModelArgs

# Provider -> OpenAI-compatible base URL. Everything talks the OpenAI wire
# protocol, so a new provider just needs its endpoint here (or an explicit
# RAG_JUDGE_BASE_URL). None = OpenAI's own default endpoint.
_OPENAI_COMPAT_BASE = {
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
    "google_genai": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

# How the judge is asked to emit structured output. `llm_factory` hardcodes the
# `json` (json_object) mode, which local servers like LM Studio reject -- they
# only accept `json_schema` or plain text. `md_json` sidesteps this entirely: it
# asks the model for JSON in a markdown block, needing no special API features,
# so it works with local models and hosted ones alike. Hence we build the
# InstructorLLM directly rather than via llm_factory.
_INSTRUCTOR_MODES = {
    "md_json": instructor.Mode.MD_JSON,
    "json": instructor.Mode.JSON,
    "json_schema": instructor.Mode.JSON_SCHEMA,
    "tools": instructor.Mode.TOOLS,
}


def build_judge(settings: Settings) -> Tuple[InstructorBaseRagasLLM, BaseRagasEmbedding]:
    """Return (judge_llm, embeddings) for the RAGAS collections metrics."""
    from openai import AsyncOpenAI

    provider = (settings.judge_provider or "openai").strip().lower()
    base_url: Optional[str] = settings.judge_base_url or _OPENAI_COMPAT_BASE.get(provider)

    # Async client: the collections metrics score via .ascore()/agenerate().
    client = AsyncOpenAI(api_key=settings.judge_api_key or None, base_url=base_url)
    mode = _INSTRUCTOR_MODES.get(settings.judge_structured_mode.strip().lower(), instructor.Mode.MD_JSON)

    llm = InstructorLLM(
        client=instructor.from_openai(client, mode=mode),
        model=settings.judge_model,
        provider="openai",
        model_args=InstructorModelArgs(
            temperature=settings.judge_temperature,
            max_tokens=settings.judge_max_tokens,
        ),
    )

    embeddings = HuggingFaceEmbeddings(
        model=settings.embedding_model,
        device=resolve_device(settings.embedding_device),
        normalize_embeddings=True,
    )
    return llm, embeddings
