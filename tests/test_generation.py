"""
tests/test_generation.py
=========================
Tests for backend selection and the offline extractive generator.

The selection logic matters more than it looks: getting it wrong either makes
a fresh clone crash on startup (the failure this design exists to prevent) or
silently sends requests to a paid API when the user did not ask for that.
"""

import pytest

from pipeline.llm_generate import (
    ExtractiveGenerator,
    build_generator,
    detect_provider,
)
from pipeline.parse_chunk import Chunk
from pipeline.rerank import RerankedChunk


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    """Never let a developer's real key change what these tests assert."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def _reranked(text: str) -> RerankedChunk:
    return RerankedChunk(
        chunk=Chunk(id="1", doc_id="d", text=text, source="a.txt", metadata={"filename": "a.txt"}),
        rerank_score=1.0,
    )


# --------------------------------------------------------------------- #
# Provider detection
# --------------------------------------------------------------------- #
def test_no_credentials_falls_back_to_extractive():
    assert detect_provider() == "extractive"


def test_anthropic_key_selects_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert detect_provider() == "anthropic"


def test_openai_key_selects_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert detect_provider() == "openai"


def test_explicit_base_url_wins_over_key_detection(monkeypatch):
    """Pointing at a local vLLM or Ollama server must not be overridden."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert detect_provider(base_url="http://localhost:11434/v1") == "openai"


def test_auto_build_without_credentials_returns_the_offline_backend():
    generator = build_generator(provider="auto")
    assert isinstance(generator, ExtractiveGenerator)
    assert generator.name == "extractive"


def test_extractive_can_be_forced_even_when_a_key_exists(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(build_generator(provider="extractive"), ExtractiveGenerator)


def test_anthropic_backend_actually_constructs(monkeypatch):
    """
    Guards SDK/dependency incompatibilities.

    The offline path never builds this client, so a constructor that raises --
    as it did when a pinned anthropic release passed `proxies=` to an httpx
    version that had removed it -- stays completely invisible until someone
    sets a key, and then fails at startup. Construction only; no network.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    generator = build_generator(provider="auto")

    assert generator.name == "anthropic"
    assert generator.model, "a default model id must be resolved when none is configured"


def test_requesting_a_provider_without_credentials_fails_loudly():
    """Better a clear error than a request guaranteed to return 401."""
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_generator(provider="anthropic")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_generator(provider="not-a-provider")


# --------------------------------------------------------------------- #
# Extractive generation
# --------------------------------------------------------------------- #
def test_extractive_answer_quotes_only_retrieved_text():
    """The grounding guarantee: every emitted sentence exists in a chunk."""
    source = (
        "PagedAttention manages the KV cache in fixed-size blocks. "
        "This removes the fragmentation that wastes GPU memory."
    )
    answer = ExtractiveGenerator().generate("", query="How does PagedAttention save memory?",
                                            chunks=[_reranked(source)])

    quoted = answer.replace("[1]", "").strip()
    for sentence in quoted.split(". "):
        assert sentence.strip(". ") in source


def test_extractive_answer_carries_citation_markers():
    answer = ExtractiveGenerator().generate(
        "", query="what is chunking?",
        chunks=[_reranked("Chunking splits documents into smaller passages before embedding them.")],
    )
    assert "[1]" in answer


def test_extractive_skips_chunks_that_share_nothing_with_the_query():
    """A chunk can survive retrieval and still be irrelevant; quoting it pads
    the answer with confident-looking noise."""
    relevant = _reranked("PagedAttention manages the KV cache in fixed-size memory blocks.")
    irrelevant = _reranked("Sourdough starters need regular feeding to remain active and healthy.")

    answer = ExtractiveGenerator().generate("", query="How does PagedAttention manage memory?",
                                            chunks=[relevant, irrelevant])

    assert "PagedAttention" in answer
    assert "Sourdough" not in answer


def test_extractive_still_answers_when_nothing_matches_lexically():
    answer = ExtractiveGenerator().generate("", query="zzz qqq",
                                            chunks=[_reranked("Some unrelated but real content here.")])
    assert answer.strip()


def test_extractive_normalises_source_line_wrapping():
    """Chunks inherit the source file's wrapping; answers should read as prose."""
    answer = ExtractiveGenerator().generate(
        "", query="what does it do?",
        chunks=[_reranked("This sentence does\nsomething useful across\nthree wrapped lines.")],
    )
    assert "\n" not in answer.replace("[1]", "").strip()


def test_extractive_handles_no_retrieved_chunks():
    answer = ExtractiveGenerator().generate("", query="anything", chunks=[])
    assert "no relevant passages" in answer.lower()


def test_extractive_respects_its_chunk_budget():
    chunks = [_reranked(f"Memory management technique number {i} is useful.") for i in range(10)]
    answer = ExtractiveGenerator(max_chunks=2).generate("", query="memory management", chunks=chunks)

    assert "[2]" in answer
    assert "[3]" not in answer
