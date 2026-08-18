"""
tests/test_integration.py
==========================
End-to-end tests that load real transformer models.

Marked `slow` because they download and run model weights. Run everything with
`pytest`, or skip these with `pytest -m "not slow"` during rapid iteration.

These exist because every other test in the suite uses fakes somewhere. A
pipeline whose parts each pass in isolation can still be wired together wrong,
and the wiring is what the user actually experiences.
"""

import pytest

from config import Settings
from pipeline.orchestrator import RAGPipeline

pytestmark = pytest.mark.slow


CORPUS = "./data/sample_docs"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """One pipeline for the module: loading these models twice is wasteful."""
    config = Settings(
        index_dir=str(tmp_path_factory.mktemp("index")),
        persist_indexes=False,
        llm_provider="extractive",   # never call a paid API from the test suite
    )
    rag = RAGPipeline(config)
    rag.ingest(CORPUS, persist=False)
    return rag


def test_ingest_produces_chunks(pipeline):
    assert len(pipeline.vector_db) > 50
    assert len(pipeline.keyword_index) == len(pipeline.vector_db)


def test_end_to_end_query_is_grounded_and_cited(pipeline):
    result = pipeline.query("How does PagedAttention reduce GPU memory waste?")

    assert result.answer.strip()
    assert result.citations, "a grounded answer must cite at least one source"
    assert result.backend == "extractive"
    # The vLLM document is the one that actually answers this.
    assert any("vllm" in c.filename for c in result.citations)


def test_every_stage_is_timed(pipeline):
    timings = pipeline.query("What is hybrid search?").timings_ms

    assert {"query_rewrite", "retrieval", "rerank", "generation", "total"} <= set(timings)
    assert timings["total"] > 0


def test_citations_only_reference_chunks_that_were_retrieved(pipeline):
    """The auditability guarantee: a citation can never point at a fabricated source."""
    result = pipeline.query("Why is reranking useful?")

    retrieved = {r.chunk.id for r in pipeline.retrieve("Why is reranking useful?")}
    assert all(c.chunk_id in retrieved for c in result.citations)


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_all_retrieval_modes_return_results(pipeline, mode):
    assert pipeline.retrieve("What is BM25?", mode=mode)


def test_retrieval_finds_the_document_that_answers_the_question(pipeline):
    results = pipeline.retrieve("What do the k1 and b parameters control in BM25?", top_k=5)
    sources = {r.chunk.metadata.get("filename") for r in results}

    assert "bm25_keyword_search.txt" in sources


def test_ingesting_the_same_corpus_twice_is_a_no_op(pipeline):
    """Reachable in production: POST /ingest twice would otherwise double the corpus."""
    before = len(pipeline.vector_db)

    pipeline.ingest(CORPUS, persist=False)

    assert len(pipeline.vector_db) == before
    assert len(pipeline.keyword_index) == before, "the two indexes must stay in agreement"


def test_deleting_a_document_removes_it_from_the_index(tmp_path):
    """A file removed from disk must stop being retrievable, not linger."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "keep.txt").write_text("Reciprocal rank fusion merges ranked lists.", encoding="utf-8")
    (corpus / "remove.txt").write_text("Sourdough starters need regular feeding.", encoding="utf-8")

    rag = RAGPipeline(Settings(index_dir=str(tmp_path / "idx"), persist_indexes=False,
                               llm_provider="extractive"))
    rag.ingest(str(corpus), persist=False)
    assert len(rag.vector_db) == 2

    (corpus / "remove.txt").unlink()
    rag.ingest(str(corpus), persist=False)

    assert len(rag.vector_db) == 1
    assert len(rag.keyword_index) == 1
    sources = {c.metadata["filename"] for c in rag.vector_db._chunks}
    assert sources == {"keep.txt"}


def test_deletion_is_scoped_to_the_ingested_directory(tmp_path):
    """Re-ingesting one directory must not evict documents from another."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "one.txt").write_text("Cross-encoders rerank candidate passages.", encoding="utf-8")
    (second / "two.txt").write_text("BM25 rewards rare exact terms.", encoding="utf-8")

    rag = RAGPipeline(Settings(index_dir=str(tmp_path / "idx"), persist_indexes=False,
                               llm_provider="extractive"))
    rag.ingest(str(first), persist=False)
    rag.ingest(str(second), persist=False)
    assert len(rag.vector_db) == 2

    # Re-ingesting only the first directory must leave the second alone.
    rag.ingest(str(first), persist=False)

    assert len(rag.vector_db) == 2
    assert {c.metadata["filename"] for c in rag.vector_db._chunks} == {"one.txt", "two.txt"}


def test_persisted_index_round_trips(tmp_path):
    config = Settings(index_dir=str(tmp_path / "idx"), persist_indexes=True, llm_provider="extractive")

    built = RAGPipeline(config)
    chunk_count = built.ingest(CORPUS)

    reloaded = RAGPipeline(config)
    assert reloaded.load_or_ingest(CORPUS) == chunk_count
    assert reloaded.query("What is RAG?").citations


def test_a_changed_corpus_invalidates_the_persisted_index(tmp_path):
    """Adding a document must not leave the service answering from a stale index."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text("Reciprocal rank fusion merges ranked lists.", encoding="utf-8")

    config = Settings(index_dir=str(tmp_path / "idx"), persist_indexes=True, llm_provider="extractive")
    RAGPipeline(config).ingest(str(corpus))

    (corpus / "two.txt").write_text("Cross-encoders rerank candidate passages.", encoding="utf-8")

    refreshed = RAGPipeline(config)
    refreshed.load_or_ingest(str(corpus))

    assert any(
        "Cross-encoders" in r.chunk.text
        for r in refreshed.retrieve("What do cross-encoders do?", top_k=5)
    )
