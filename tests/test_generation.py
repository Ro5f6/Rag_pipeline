"""
tests/test_generation.py
=========================
Tests for the LangChain-backed generation layer.

Two things matter here: that ChatModelGenerator faithfully turns a chat model's
response into answer text, and that build_generator maps configuration onto the
right provider (and fails clearly when it can't). No test makes a network call
-- the wrapper is exercised with a fake chat model, and provider construction
is checked without ever invoking the model.
"""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from pipeline.llm_generate import (
    ChatModelGenerator,
    _resolve_provider,
    build_generator,
)


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    """Never let a developer's real key change what these tests assert."""
    for var in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _generator(reply: str) -> ChatModelGenerator:
    fake = GenericFakeChatModel(messages=iter([reply]))
    return ChatModelGenerator(fake, provider="fake", model="fake-1")


# --------------------------------------------------------------------- #
# ChatModelGenerator: response -> answer text
# --------------------------------------------------------------------- #
def test_generator_returns_the_models_text():
    gen = _generator("PagedAttention pages the KV cache. [1]")
    assert gen.generate("prompt", query="q", chunks=[]) == "PagedAttention pages the KV cache. [1]"


def test_generator_preserves_citation_markers():
    answer = _generator("Grounded claim. [1] Another. [2]").generate("p", query="q", chunks=[])
    assert "[1]" in answer and "[2]" in answer


def test_generator_strips_surrounding_whitespace():
    assert _generator("  spaced answer  \n").generate("p", query="q", chunks=[]) == "spaced answer"


def test_generator_strips_reasoning_think_blocks():
    """Reasoning models leak <think>...</think>; only the answer should survive."""
    gen = _generator("<think>Let me work through this step by step.</think>\n\nPagedAttention pages the KV cache. [1]")
    assert gen.generate("p", query="q", chunks=[]) == "PagedAttention pages the KV cache. [1]"


def test_generator_handles_unpaired_think_close():
    gen = _generator("reasoning that lost its opening tag</think>The real answer.")
    assert gen.generate("p", query="q", chunks=[]) == "The real answer."


def test_generator_flattens_block_style_content():
    """Some providers return content as a list of blocks rather than a string."""

    class _BlockModel:
        def invoke(self, _prompt):
            return AIMessage(content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}])

    gen = ChatModelGenerator(_BlockModel(), provider="fake", model="fake-1")
    assert gen.generate("p", query="q", chunks=[]) == "part one part two"


def test_generator_reports_its_provider_and_model():
    gen = _generator("x")
    assert gen.name == "fake"
    assert gen.model == "fake-1"


# --------------------------------------------------------------------- #
# Provider resolution
# --------------------------------------------------------------------- #
def test_provider_ids_pass_through_normalized():
    assert _resolve_provider("groq") == "groq"
    assert _resolve_provider("google_genai") == "google_genai"
    assert _resolve_provider("  OpenAI ") == "openai"


def test_auto_provider_defers_inference_to_the_model_name():
    assert _resolve_provider("auto") is None
    assert _resolve_provider("") is None


# --------------------------------------------------------------------- #
# build_generator
# --------------------------------------------------------------------- #
def test_missing_model_fails_loudly():
    with pytest.raises(ValueError, match="RAG_LLM_MODEL"):
        build_generator(provider="groq", model="")


def test_groq_backend_actually_constructs():
    """
    Guards SDK/dependency incompatibilities: init_chat_model must be able to
    build the provider client. Construction only -- the model is never invoked,
    so no network and no key are required.
    """
    gen = build_generator(provider="groq", model="llama-3.3-70b-versatile", api_key="dummy-key")
    assert gen.name == "groq"
    assert gen.model == "llama-3.3-70b-versatile"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Could not initialise"):
        build_generator(provider="not-a-provider", model="whatever")
