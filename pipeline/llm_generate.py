"""
pipeline/llm_generate.py
==========================
FLOWCHART BLOCK: "LLM generate"

Responsibility: turn the formatted prompt (query + retrieved context) into a
grounded answer. This is the only block in the pipeline that talks to a model
provider -- everything before it is retrieval engineering, everything after it
is post-processing.

DESIGN: the provider is a swappable detail, not a dependency.
--------------------------------------------------------------------------
Nothing upstream or downstream of this file knows which model answers the
question. Every backend implements exactly one method:

    generate(prompt, *, query, chunks) -> str

so switching from a hosted API to a self-hosted GPU server is a config change,
not a code change. Four backends ship here:

  AnthropicGenerator        Claude, via the Anthropic Messages API.
  OpenAICompatibleGenerator Any OpenAI-shaped /chat/completions endpoint. One
                            class covers OpenAI, a self-hosted vLLM or TGI
                            server, Ollama, Together, Groq, OpenRouter, and
                            most gateways -- they differ only by base_url.
  GoogleGenerator           Gemini, via the google-genai SDK.
  ExtractiveGenerator       No API, no key, no network, no cost.

Adding a fourth (Bedrock, Vertex, a bespoke internal endpoint) means writing
one class with one method and registering it in _BACKENDS below. That is the
whole point of isolating generation into its own block.

WHY THE EXTRACTIVE BACKEND IS NOT A TOY
--------------------------------------------------------------------------
A RAG system's answer quality is dominated by *retrieval*, not by the final
generation call. Keeping a backend that needs no credentials means the entire
retrieval stack -- chunking, embedding, hybrid search, fusion, reranking,
citation mapping -- stays runnable and testable by anyone who clones this
repo, and stays runnable in CI where no secret exists. It never invents text:
it only quotes sentences that are actually present in the retrieved chunks, so
its answers are grounded by construction. It is a weaker writer than an LLM,
not a less trustworthy one.

The active backend is reported all the way out to the API response, so an
answer is never silently mistaken for something it isn't.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


class Generator(Protocol):
    """The one interface every generation backend implements."""

    name: str
    model: str

    def generate(self, prompt: str, *, query: str, chunks: Sequence) -> str:
        ...


def _retry(call: Callable[[], str], attempts: int, is_retryable: Callable[[Exception], bool], label: str) -> str:
    """Shared bounded-backoff retry, so each provider class stays thin."""
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retryable
            if not is_retryable(exc):
                raise
            last_error = exc
            backoff = 2**attempt
            logger.warning(
                "%s call failed (attempt %d/%d): %s. Retrying in %ds.",
                label, attempt + 1, attempts, exc, backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(f"{label} generation failed after {attempts} attempts") from last_error


# --------------------------------------------------------------------- #
# Backend: Anthropic
# --------------------------------------------------------------------- #
class AnthropicGenerator:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        api_key: str = "",
        base_url: str = "",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        import anthropic  # imported lazily: only this backend needs the SDK

        self._sdk = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
            **({"base_url": base_url} if base_url else {}),
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (self._sdk.RateLimitError, self._sdk.APIConnectionError)):
            return True
        # A 4xx that isn't rate limiting (bad key, unknown model) will never
        # fix itself on retry -- surface it immediately.
        return isinstance(exc, self._sdk.APIStatusError) and exc.status_code >= 500

    def generate(self, prompt: str, *, query: str = "", chunks: Sequence = ()) -> str:
        def call() -> str:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text")

        return _retry(call, self.max_retries, self._is_retryable, "Anthropic")


# --------------------------------------------------------------------- #
# Backend: anything speaking the OpenAI /chat/completions shape
# --------------------------------------------------------------------- #
class OpenAICompatibleGenerator:
    """
    One class for every OpenAI-shaped endpoint. Point base_url at whatever is
    serving the model:

        OpenAI          (default, no base_url needed)
        Self-hosted     http://localhost:8000/v1     (vLLM, TGI, LMStudio)
        Ollama          http://localhost:11434/v1
        Groq            https://api.groq.com/openai/v1
        Together        https://api.together.xyz/v1
        OpenRouter      https://openrouter.ai/api/v1

    Self-hosted servers usually ignore the key entirely, so a placeholder is
    sent to satisfy the client rather than requiring the user to invent one.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        api_key: str = "",
        base_url: str = "",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        from openai import OpenAI  # lazy: only this backend needs the SDK

        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-required",
            timeout=timeout,
            **({"base_url": base_url} if base_url else {}),
        )

    def _is_retryable(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            # Connection-level failures carry no status code.
            return "connection" in type(exc).__name__.lower()
        return status == 429 or status >= 500

    def generate(self, prompt: str, *, query: str = "", chunks: Sequence = ()) -> str:
        def call() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        return _retry(call, self.max_retries, self._is_retryable, "OpenAI-compatible")


# --------------------------------------------------------------------- #
# Backend: Google Gemini
# --------------------------------------------------------------------- #
class GoogleGenerator:
    """
    Google Gemini via the google-genai SDK.

    The key is read from GOOGLE_API_KEY (or GEMINI_API_KEY) when not passed
    explicitly. base_url is honoured for gateways or a Vertex-backed endpoint.
    """

    name = "google"

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        max_tokens: int = 1024,
        api_key: str = "",
        base_url: str = "",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        from google import genai  # imported lazily: only this backend needs the SDK
        from google.genai import types

        self._types = types
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                base_url=base_url or None,
                timeout=int(timeout * 1000) if timeout else None,  # google-genai expects milliseconds
            ),
        )

    def _is_retryable(self, exc: Exception) -> bool:
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            # 429 (rate limit) and 5xx are transient; a 4xx like a bad key or
            # unknown model will never fix itself on retry.
            return code == 429 or (isinstance(code, int) and code >= 500)
        # Connection-level failures carry no API status code.
        return "connection" in type(exc).__name__.lower()

    def generate(self, prompt: str, *, query: str = "", chunks: Sequence = ()) -> str:
        def call() -> str:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(max_output_tokens=self.max_tokens),
            )
            return response.text or ""

        return _retry(call, self.max_retries, self._is_retryable, "Google")


# --------------------------------------------------------------------- #
# Backend: extractive (no API key, no network, no cost)
# --------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "in", "is", "it",
    "its", "of", "on", "or", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _content_terms(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _normalize(text: str) -> str:
    """Chunks carry the source file's line wrapping; answers should not."""
    return re.sub(r"\s+", " ", text).strip()


def _best_sentences(text: str, query_terms: set, limit: int = 2) -> List[tuple]:
    """
    Pick the sentences within a chunk that most directly address the query.

    Returns (overlap, sentence) pairs, ordered best-first, so the caller can
    tell a genuinely relevant chunk from one that merely survived retrieval.
    """
    sentences = [_normalize(s) for s in _SENTENCE_SPLIT.split(text)]
    sentences = [s for s in sentences if len(s) > 25]
    if not sentences:
        stripped = _normalize(text)
        return [(0, stripped)] if stripped else []

    scored = []
    for position, sentence in enumerate(sentences):
        terms = _content_terms(sentence)
        overlap = len(query_terms & terms)
        # Normalising by length stops a long rambling sentence from winning
        # purely by containing more words; the position tiebreak reflects that
        # chunks tend to lead with their topic sentence.
        density = overlap / ((len(terms) ** 0.5) or 1)
        scored.append((overlap, density, -position, sentence))

    scored.sort(reverse=True)

    # Restore document order among the winners: quoting sentences out of their
    # original sequence reads as incoherent even when each one is relevant.
    winners = sorted(scored[:limit], key=lambda s: -s[2])
    return [(overlap, sentence) for overlap, _, _, sentence in winners]


class ExtractiveGenerator:
    """
    Composes an answer purely by quoting retrieved chunks.

    It cannot hallucinate: every sentence it emits appears verbatim in a chunk,
    and carries the citation marker of the chunk it came from.
    """

    name = "extractive"
    model = "none (extractive)"

    def __init__(self, max_chunks: int = 3, sentences_per_chunk: int = 2):
        self.max_chunks = max_chunks
        self.sentences_per_chunk = sentences_per_chunk

    def generate(self, prompt: str, *, query: str = "", chunks: Sequence = ()) -> str:
        if not chunks:
            return "The knowledge base returned no relevant passages for this question."

        query_terms = _content_terms(query)
        matched: List[str] = []
        fallback: List[str] = []

        for index, reranked in enumerate(chunks[: self.max_chunks], start=1):
            text = getattr(reranked, "chunk", reranked).text
            scored = _best_sentences(text, query_terms, self.sentences_per_chunk)
            if not scored:
                continue

            passage = f"{' '.join(s for _, s in scored)} [{index}]"
            # A chunk can survive retrieval on semantic similarity alone while
            # sharing no meaningful term with the question. Quoting it pads the
            # answer with confident-looking irrelevance, so it is only used if
            # nothing better matched.
            (matched if any(overlap > 0 for overlap, _ in scored) else fallback).append(passage)

        chosen = matched or fallback[:1]
        if not chosen:
            return "The retrieved context does not contain an answer to this question."

        return "\n\n".join(chosen)


# --------------------------------------------------------------------- #
# Backend registry & selection
# --------------------------------------------------------------------- #
_BACKENDS: Dict[str, type] = {
    "anthropic": AnthropicGenerator,
    "openai": OpenAICompatibleGenerator,
    "google": GoogleGenerator,
    "extractive": ExtractiveGenerator,
}

# Sensible default model per provider, used when llm_model is left blank.
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "google": "gemini-3.5-flash",
}

# Which env var implies "this provider is configured and ready".
_PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def detect_provider(api_key: str = "", base_url: str = "") -> str:
    """
    Resolve provider='auto'.

    An explicit base_url means the user is pointing at a specific server (a
    local vLLM, Ollama, a gateway), so honour that first. Otherwise pick
    whichever provider has credentials present, and fall back to extractive so
    the pipeline always runs.
    """
    if base_url:
        return "openai"
    if api_key or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "google"
    return "extractive"


def build_generator(
    provider: str = "auto",
    model: str = "",
    max_tokens: int = 1024,
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
) -> Generator:
    """
    Build the configured generation backend.

    provider='auto' picks a real model when credentials exist and the
    extractive backend when they don't, so a fresh clone always runs.
    """
    resolved = detect_provider(api_key, base_url) if provider == "auto" else provider

    if resolved not in _BACKENDS:
        raise ValueError(f"Unknown provider {resolved!r}. Available: {sorted(_BACKENDS)}")

    if resolved == "extractive":
        if provider == "auto":
            logger.info("No model credentials found -- using the extractive generator (offline, free).")
        return ExtractiveGenerator()

    # Fail clearly rather than firing a request that is guaranteed to 401.
    key = api_key or os.environ.get(_PROVIDER_KEY_VARS.get(resolved, ""), "")
    if provider != "auto" and not key and not base_url:
        raise ValueError(
            f"provider={resolved!r} needs {_PROVIDER_KEY_VARS[resolved]} to be set, "
            f"or a base_url pointing at a self-hosted server. "
            f"Leave provider='auto' to fall back to the offline backend instead."
        )

    chosen_model = model or _DEFAULT_MODELS.get(resolved, "")
    logger.info("Using %s backend with model %s.", resolved, chosen_model)

    return _BACKENDS[resolved](
        model=chosen_model,
        max_tokens=max_tokens,
        api_key=key,
        base_url=base_url,
        timeout=timeout,
    )
