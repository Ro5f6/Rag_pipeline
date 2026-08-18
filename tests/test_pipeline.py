"""
tests/test_pipeline.py
=======================
Block-level unit tests. Each test isolates ONE stage of the pipeline, so a
failure names the broken block rather than reporting that "RAG is wrong
somewhere".

These deliberately avoid loading transformer models: every test here runs in
milliseconds against hand-built inputs. Tests that need real model weights are
marked `slow` and live in test_integration.py.
"""

import numpy as np
import pytest

from pipeline.cite_sources import attach_citations, build_citation_map
from pipeline.context_format import build_prompt, format_context
from pipeline.documents import Document, corpus_fingerprint, load_documents_from_dir
from pipeline.hybrid_search import reciprocal_rank_fusion
from pipeline.keyword_index import KeywordIndex
from pipeline.parse_chunk import Chunk, parse_and_chunk
from pipeline.query_rewrite import rule_based_rewrite
from pipeline.rerank import RerankedChunk
from pipeline.vector_db import VectorDB


# --------------------------------------------------------------------- #
# Parse & chunk
# --------------------------------------------------------------------- #
def test_chunking_produces_real_overlap():
    """Consecutive chunks must genuinely share text, or a split definition
    becomes unretrievable from either side."""
    doc = Document(id="d1", text="abcdefghijklmnop", source="test.txt")
    chunks = parse_and_chunk([doc], chunk_size=8, chunk_overlap=3)

    assert len(chunks) > 1
    overlap_size = 3
    for previous, current in zip(chunks, chunks[1:]):
        # The tail of each chunk must literally begin the next one.
        assert previous.text[-overlap_size:] == current.text[:overlap_size]


def test_chunking_prefers_natural_boundaries():
    """Recursive splitting should break on paragraphs, not mid-sentence."""
    doc = Document(
        id="d1",
        text="First paragraph about retrieval.\n\nSecond paragraph about ranking.",
        source="test.txt",
    )
    chunks = parse_and_chunk([doc], chunk_size=40, chunk_overlap=0)

    assert [c.text for c in chunks] == [
        "First paragraph about retrieval.",
        "Second paragraph about ranking.",
    ]


def test_chunking_rejects_overlap_larger_than_chunk_size():
    """Validated on splitter construction, before any document is read."""
    doc = Document(id="d1", text="a" * 50, source="test.txt")
    with pytest.raises(ValueError):
        parse_and_chunk([doc], chunk_size=10, chunk_overlap=20)


def test_chunk_ids_are_unique_and_carry_document_metadata():
    docs = [
        Document(id="d1", text="x" * 200, source="a.txt", metadata={"filename": "a.txt"}),
        Document(id="d2", text="y" * 200, source="b.txt", metadata={"filename": "b.txt"}),
    ]
    chunks = parse_and_chunk(docs, chunk_size=50, chunk_overlap=5)

    assert len({c.id for c in chunks}) == len(chunks)
    assert all("filename" in c.metadata for c in chunks)
    assert all(c.metadata["chunk_index"] == int(c.id.split("_")[-1]) for c in chunks)


def test_empty_documents_are_skipped():
    docs = [Document(id="d1", text="   \n  ", source="empty.txt")]
    assert parse_and_chunk(docs) == []


def test_parse_and_chunk_uses_recursive_splitter(monkeypatch):
    seen = {}

    class FakeSplitter:
        def __init__(self, chunk_size, chunk_overlap, separators=None):
            seen["config"] = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

        def split_text(self, text):
            seen["text"] = text
            return ["alpha", "beta"]

    monkeypatch.setattr("pipeline.parse_chunk.RecursiveCharacterTextSplitter", FakeSplitter)

    doc = Document(id="d1", text="long narrative text", source="test.txt")
    chunks = parse_and_chunk([doc], chunk_size=20, chunk_overlap=5)

    assert seen["config"] == {"chunk_size": 20, "chunk_overlap": 5}
    assert [c.text for c in chunks] == ["alpha", "beta"]


def test_chunking_failure_is_not_swallowed(monkeypatch):
    """There is no fallback strategy by design: a silent switch would mean the
    corpus could be split two different ways depending on a transient failure,
    and the retrieval metrics would no longer describe what is running."""

    class ExplodingSplitter:
        def __init__(self, chunk_size, chunk_overlap, separators=None):
            pass

        def split_text(self, text):
            raise RuntimeError("recursive splitter failed")

    monkeypatch.setattr("pipeline.parse_chunk.RecursiveCharacterTextSplitter", ExplodingSplitter)

    doc = Document(id="d1", text="abcdefghij", source="test.txt")
    with pytest.raises(RuntimeError, match="recursive splitter failed"):
        parse_and_chunk([doc], chunk_size=4, chunk_overlap=1)


# --------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------- #
def test_document_ids_are_stable_across_loads(tmp_path):
    (tmp_path / "one.txt").write_text("hello world", encoding="utf-8")

    first = load_documents_from_dir(str(tmp_path))
    second = load_documents_from_dir(str(tmp_path))

    assert first[0].id == second[0].id, "re-ingesting the same file must not create a new id"


def test_corpus_fingerprint_changes_when_a_document_is_added(tmp_path):
    (tmp_path / "one.txt").write_text("hello", encoding="utf-8")
    before = corpus_fingerprint(str(tmp_path))

    (tmp_path / "two.txt").write_text("world", encoding="utf-8")
    after = corpus_fingerprint(str(tmp_path))

    assert before != after, "a stale index would otherwise be reused after the corpus changed"


# --------------------------------------------------------------------- #
# Vector DB
# --------------------------------------------------------------------- #
def _chunk(cid: str, text: str = "text", source: str = "a.txt") -> Chunk:
    return Chunk(id=cid, doc_id="d1", text=text, source=source, metadata={"filename": source})


def test_vector_db_returns_closest_first():
    db = VectorDB(dim=4)
    db.add(
        [_chunk("1", "chunk one"), _chunk("2", "chunk two")],
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"),
    )

    results = db.search(np.array([0.9, 0.1, 0, 0], dtype="float32"), top_k=2)

    assert [r.chunk.id for r in results] == ["1", "2"]
    assert results[0].score > results[1].score


def test_vector_db_rejects_mismatched_dimensions():
    db = VectorDB(dim=4)
    with pytest.raises(ValueError):
        db.add([_chunk("1")], np.zeros((1, 8), dtype="float32"))


def test_vector_db_handles_top_k_larger_than_corpus():
    db = VectorDB(dim=2)
    db.add([_chunk("1")], np.array([[1, 0]], dtype="float32"))

    results = db.search(np.array([1, 0], dtype="float32"), top_k=50)
    assert len(results) == 1, "FAISS pads with -1; those must not become results"


def test_empty_vector_db_returns_no_results():
    assert VectorDB(dim=4).search(np.zeros(4, dtype="float32")) == []


def test_reingesting_the_same_document_does_not_duplicate():
    """Ingestion must be idempotent, or repeated /ingest calls double the corpus."""
    db = VectorDB(dim=2)
    chunks = [_chunk("d1_0", "first"), _chunk("d1_1", "second")]
    vectors = np.array([[1, 0], [0, 1]], dtype="float32")

    db.add(chunks, vectors)
    db.add(chunks, vectors)

    assert len(db) == 2
    assert db.index.ntotal == 2, "the FAISS index must not drift from the chunk list"


def test_reingesting_an_edited_document_replaces_its_chunks():
    """The failure mode a naive id-skip would cause: edits silently ignored."""
    db = VectorDB(dim=2)
    db.add([_chunk("d1_0", "original text")], np.array([[1, 0]], dtype="float32"))
    db.add([_chunk("d1_0", "edited text")], np.array([[0, 1]], dtype="float32"))

    assert len(db) == 1
    assert db._chunks[0].text == "edited text"


def test_replacing_one_document_leaves_the_others_intact():
    db = VectorDB(dim=2)
    keep = Chunk(id="d2_0", doc_id="d2", text="untouched", source="b.txt", metadata={})
    db.add([_chunk("d1_0", "old"), keep], np.array([[1, 0], [0, 1]], dtype="float32"))

    db.add([_chunk("d1_0", "new")], np.array([[1, 0]], dtype="float32"))

    texts = {c.text for c in db._chunks}
    assert texts == {"untouched", "new"}
    assert db.index.ntotal == 2


def test_keyword_index_replaces_documents_rather_than_appending():
    idx = KeywordIndex()
    idx.add([_chunk("d1_0", "PagedAttention manages memory")])
    idx.add([_chunk("d1_0", "PagedAttention manages memory")])

    assert len(idx) == 1


def test_keyword_index_picks_up_edited_content():
    idx = KeywordIndex()
    idx.add([_chunk("d1_0", "gardening tips for beginners")])
    idx.add([_chunk("d1_0", "PagedAttention manages memory")])

    assert len(idx) == 1
    assert idx.search("PagedAttention")
    assert not idx.search("gardening")


def test_remove_documents_drops_only_the_named_documents():
    db = VectorDB(dim=2)
    keep = Chunk(id="d2_0", doc_id="d2", text="keep me", source="b.txt", metadata={})
    db.add([_chunk("d1_0", "drop me"), keep], np.array([[1, 0], [0, 1]], dtype="float32"))

    removed = db.remove_documents({"d1"})

    assert removed == 1
    assert [c.text for c in db._chunks] == ["keep me"]
    assert db.index.ntotal == 1


def test_remove_documents_can_empty_the_index():
    db = VectorDB(dim=2)
    db.add([_chunk("d1_0", "only one")], np.array([[1, 0]], dtype="float32"))

    db.remove_documents({"d1"})

    assert len(db) == 0
    assert db.search(np.array([1, 0], dtype="float32")) == []


def test_remove_documents_ignores_unknown_ids():
    db = VectorDB(dim=2)
    db.add([_chunk("d1_0", "text")], np.array([[1, 0]], dtype="float32"))

    assert db.remove_documents({"never-indexed"}) == 0
    assert len(db) == 1


def test_keyword_index_remove_documents_rebuilds_bm25():
    idx = KeywordIndex()
    keep = Chunk(id="d2_0", doc_id="d2", text="gardening tips", source="b.txt", metadata={})
    idx.add([_chunk("d1_0", "PagedAttention manages memory"), keep])

    assert idx.remove_documents({"d1"}) == 1
    assert not idx.search("PagedAttention")
    assert idx.search("gardening")


def test_keyword_index_survives_being_emptied():
    """BM25Okapi cannot be built over an empty corpus; search must still work."""
    idx = KeywordIndex()
    idx.add([_chunk("d1_0", "some text")])

    idx.remove_documents({"d1"})

    assert len(idx) == 0
    assert idx.search("anything") == []


def test_vector_db_survives_a_save_load_round_trip(tmp_path):
    db = VectorDB(dim=2)
    db.add([_chunk("1", "first"), _chunk("2", "second")], np.array([[1, 0], [0, 1]], dtype="float32"))
    db.save(str(tmp_path))

    restored = VectorDB.load(str(tmp_path))

    assert len(restored) == 2
    assert restored.dim == 2
    assert restored.search(np.array([1, 0], dtype="float32"), top_k=1)[0].chunk.text == "first"


# --------------------------------------------------------------------- #
# Keyword index
# --------------------------------------------------------------------- #
def test_keyword_index_finds_exact_rare_term():
    idx = KeywordIndex()
    idx.add([
        _chunk("1", "PagedAttention manages memory in blocks"),
        _chunk("2", "completely unrelated sentence about gardening"),
    ])

    results = idx.search("PagedAttention")

    assert results and results[0].chunk.id == "1"


def test_keyword_index_returns_nothing_for_an_empty_query():
    idx = KeywordIndex()
    idx.add([_chunk("1", "some text")])
    assert idx.search("   ") == []


def test_empty_keyword_index_returns_no_results():
    assert KeywordIndex().search("anything") == []


def test_keyword_index_survives_a_save_load_round_trip(tmp_path):
    idx = KeywordIndex()
    idx.add([_chunk("1", "PagedAttention blocks"), _chunk("2", "gardening tips")])
    idx.save(str(tmp_path))

    restored = KeywordIndex.load(str(tmp_path))

    assert len(restored) == 2
    assert restored.search("PagedAttention")[0].chunk.id == "1"


# --------------------------------------------------------------------- #
# Hybrid search / reciprocal rank fusion
# --------------------------------------------------------------------- #
class _Result:
    def __init__(self, chunk):
        self.chunk = chunk


def test_rrf_boosts_a_chunk_both_retrievers_returned():
    c1, c2, c3 = _chunk("1"), _chunk("2"), _chunk("3")

    fused = reciprocal_rank_fusion(
        vector_results=[_Result(c2), _Result(c1)],   # c1 second here
        keyword_results=[_Result(c3), _Result(c1)],  # c1 second here too
    )

    # c1 is first in neither list, but is the only chunk both retrievers found.
    assert fused[0].chunk.id == "1"


def test_rrf_records_the_rank_from_each_retriever():
    c1 = _chunk("1")
    fused = reciprocal_rank_fusion([_Result(c1)], [])

    assert fused[0].vector_rank == 1
    assert fused[0].keyword_rank is None


def test_rrf_score_matches_the_formula():
    c1 = _chunk("1")
    fused = reciprocal_rank_fusion([_Result(c1)], [_Result(c1)], k=60)

    assert fused[0].rrf_score == pytest.approx(2 * (1 / 61))


def test_rrf_truncates_to_top_k():
    results = [_Result(_chunk(str(i))) for i in range(10)]
    assert len(reciprocal_rank_fusion(results, [], top_k=3)) == 3


def test_rrf_ordering_is_deterministic_for_tied_scores():
    """Reproducible evaluation runs depend on ties breaking the same way."""
    a = reciprocal_rank_fusion([_Result(_chunk("b")), _Result(_chunk("a"))], [])
    b = reciprocal_rank_fusion([_Result(_chunk("b")), _Result(_chunk("a"))], [])
    assert [c.chunk.id for c in a] == [c.chunk.id for c in b]


# --------------------------------------------------------------------- #
# Query rewrite
# --------------------------------------------------------------------- #
def test_rule_based_rewrite_expands_a_known_acronym():
    assert "retrieval augmented generation" in rule_based_rewrite("what is RAG?")


def test_rule_based_rewrite_does_not_duplicate_an_existing_expansion():
    rewritten = rule_based_rewrite("rag means retrieval augmented generation")
    assert rewritten.count("retrieval augmented generation") == 1


def test_rule_based_rewrite_collapses_whitespace():
    assert rule_based_rewrite("  what   is\n\nthis  ") == "what is this"


# --------------------------------------------------------------------- #
# Context format
# --------------------------------------------------------------------- #
def _reranked(cid: str, text: str, filename: str, score: float) -> RerankedChunk:
    return RerankedChunk(chunk=_chunk(cid, text, filename), rerank_score=score)


def test_context_format_numbers_sources_in_rank_order():
    context = format_context([
        _reranked("1", "first", "a.txt", 1.0),
        _reranked("2", "second", "b.txt", 0.5),
    ])

    assert context.index("[1]") < context.index("[2]")
    assert "a.txt" in context and "b.txt" in context


def test_build_prompt_instructs_the_model_to_ground_and_cite():
    prompt = build_prompt("What is X?", "[1] (source: a.txt)\nsome context")

    assert "What is X?" in prompt
    assert "some context" in prompt
    # Grounding and permission-to-fail are the two instructions that make
    # citation possible downstream; losing either silently breaks auditability.
    assert "only" in prompt.lower()
    assert "cite" in prompt.lower()


# --------------------------------------------------------------------- #
# Cite sources
# --------------------------------------------------------------------- #
def test_citation_map_numbers_chunks_from_one():
    citation_map = build_citation_map([_reranked("1", "a", "a.txt", 1.0), _reranked("2", "b", "b.txt", 0.5)])
    assert sorted(citation_map) == ["[1]", "[2]"]


def test_only_referenced_citations_are_returned():
    chunks = [_reranked("1", "first", "a.txt", 1.0), _reranked("2", "second", "b.txt", 0.5)]

    final = attach_citations("The answer is X [1].", chunks)

    assert [c.marker for c in final.citations] == ["[1]"]
    assert final.citations[0].filename == "a.txt"


def test_an_answer_citing_nothing_returns_no_citations():
    final = attach_citations("I could not find this.", [_reranked("1", "first", "a.txt", 1.0)])
    assert final.citations == []
