"""
pipeline/keyword_index.py
==========================
FLOWCHART BLOCK: "Keyword index"

Responsibility: lexical (exact-term) search using BM25 -- the classic ranking
function behind search engines predating neural retrieval.

Why keep this alongside a vector DB instead of embedding everything?
Semantic search is excellent at paraphrase matching and terrible at rare exact
tokens: product SKUs, error codes, version numbers, proper nouns, acronyms the
embedding model never saw enough of to place meaningfully in vector space. An
embedding of "ORA-01555" is close to nothing useful. BM25 nails it, because it
scores direct term overlap weighted by how rare the term is across the corpus
(inverse document frequency) and how concentrated it is in a given document
(term frequency, with saturation).

Vector search and BM25 fail on *different* queries, which is exactly why
fusing them (hybrid_search.py) beats either alone -- and why the eval harness
measures all three configurations separately.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set

from rank_bm25 import BM25Okapi

from pipeline.parse_chunk import Chunk

_CHUNKS_FILE = "keyword_chunks.json"


@dataclass
class ScoredChunkBM25:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> List[str]:
    # Naive whitespace + lowercase tokenizer. A production corpus with heavy
    # punctuation or multilingual text wants a real tokenizer (spaCy, nltk)
    # and a stemmer; this is the one place that would need to change.
    return text.lower().split()


class KeywordIndex:
    def __init__(self):
        self._chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None

    def add(self, chunks: List[Chunk]) -> int:
        """
        Index the given chunks, replacing any document already present.

        Matches VectorDB.add()'s replace-by-document semantics so the two
        indexes can never disagree about what the corpus contains -- a
        divergence would show up as hybrid search fusing results from two
        different versions of the same document.

        Note the asymmetry with VectorDB in *cost*: rank_bm25 recomputes corpus
        statistics (IDF over every document) from scratch, so the whole index
        is rebuilt on every add regardless. That is fine for batch ingestion
        and wrong for a high-write workload -- the point at which you'd move to
        Elasticsearch/OpenSearch, which index incrementally.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        incoming_docs = {c.doc_id for c in chunks}
        retained = [c for c in self._chunks if c.doc_id not in incoming_docs]

        self._chunks = retained + list(chunks)
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks])
        return len(chunks)

    def remove_documents(self, doc_ids: Set[str]) -> int:
        """
        Drop every chunk belonging to the given documents.

        Mirrors VectorDB.remove_documents so a deletion applied to one index is
        always applied to the other. If the two ever disagreed, hybrid search
        would fuse a result set where one retriever still knows about a
        document the other has forgotten.

        Returns the number of chunks removed.
        """
        if not doc_ids:
            return 0

        retained = [c for c in self._chunks if c.doc_id not in doc_ids]
        removed = len(self._chunks) - len(retained)
        if not removed:
            return 0

        self._chunks = retained
        # BM25Okapi cannot be built over an empty corpus; None is the same
        # state a fresh index starts in, and search() already handles it.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks]) if self._chunks else None

        return removed

    def search(self, query: str, top_k: int = 20) -> List[ScoredChunkBM25]:
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)

        # BM25 assigns zero to every document when each query term appears in
        # every document (IDF collapses to zero on a tiny corpus). Rather than
        # returning nothing, fall back to raw term-overlap counting so hybrid
        # fusion still receives a usable lexical ranking.
        if not any(score > 0 for _, score in ranked):
            ranked = sorted(
                (
                    (chunk, float(sum(1 for term in tokenized_query if term in _tokenize(chunk.text))))
                    for chunk in self._chunks
                ),
                key=lambda x: x[1],
                reverse=True,
            )

        return [ScoredChunkBM25(chunk=c, score=float(s)) for c, s in ranked[:top_k] if s > 0]

    # ------------------------------------------------------------------ #
    # Persistence: only the corpus is stored; BM25 stats rebuild on load
    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / _CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str) -> "KeywordIndex":
        chunks = [
            Chunk(**c)
            for c in json.loads((Path(directory) / _CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        index = cls()
        index.add(chunks)
        return index

    @staticmethod
    def exists(directory: str) -> bool:
        return (Path(directory) / _CHUNKS_FILE).exists()

    def __len__(self) -> int:
        return len(self._chunks)
