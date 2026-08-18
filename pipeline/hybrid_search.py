"""
pipeline/hybrid_search.py
===========================
FLOWCHART BLOCK: "Hybrid search"

Responsibility: this is where the three upstream arrows (Query rewrite ->,
Vector DB ->, Keyword index ->) actually merge into one ranked candidate list.

The hard part is that the two retrievers produce scores on incomparable
scales. Cosine similarity is bounded in [-1, 1] and tends to cluster tightly
around 0.3-0.7 for real text; BM25 is unbounded, corpus-dependent, and can
return 14.2 on one query and 0.8 on the next. Normalising them (min-max,
z-score) is possible but fragile: the normalisation constants shift with every
query and every corpus update.

Reciprocal Rank Fusion sidesteps the problem entirely by discarding raw scores
and using only each result's *rank position* within its own list:

    RRF(chunk) = sum over retrievers r containing the chunk of  1 / (k + rank_r)

k (conventionally 60) dampens the advantage of the very top ranks, so a single
retriever cannot dominate the fused ordering. A chunk that both retrievers
rank highly outscores a chunk that only one of them loves -- which is exactly
the agreement signal hybrid search exists to capture.

RRF is also parameter-light and corpus-independent, meaning it does not need
retuning when the knowledge base changes. That property is worth more in
production than the extra point or two of nDCG a tuned weighted-sum might buy.

`mode` exposes each retriever in isolation so the evaluation harness can
measure what fusion is actually contributing (see evaluation/run_eval.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pipeline.embed import Embedder
from pipeline.keyword_index import KeywordIndex
from pipeline.parse_chunk import Chunk
from pipeline.vector_db import VectorDB


@dataclass
class FusedChunk:
    chunk: Chunk
    rrf_score: float
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None


def reciprocal_rank_fusion(
    vector_results: List,      # List[ScoredChunk]
    keyword_results: List,     # List[ScoredChunkBM25]
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[FusedChunk]:
    scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, Chunk] = {}
    ranks: Dict[str, Dict[str, int]] = {}   # chunk_id -> {"vector": rank, "keyword": rank}

    for source, results in (("vector", vector_results), ("keyword", keyword_results)):
        for rank, result in enumerate(results, start=1):
            cid = result.chunk.id
            chunk_lookup[cid] = result.chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[source] = rank

    fused = [
        FusedChunk(
            chunk=chunk_lookup[cid],
            rrf_score=score,
            vector_rank=ranks[cid].get("vector"),
            keyword_rank=ranks[cid].get("keyword"),
        )
        for cid, score in scores.items()
    ]
    # Ties are broken by chunk id so the ordering is deterministic -- without
    # this, evaluation runs would not be reproducible.
    fused.sort(key=lambda x: (-x.rrf_score, x.chunk.id))

    return fused[:top_k] if top_k else fused


def hybrid_search(
    query: str,
    vector_db: VectorDB,
    keyword_index: KeywordIndex,
    embedder: Embedder,
    top_k: int = 20,
    rrf_k: int = 60,
    mode: str = "hybrid",
) -> List[FusedChunk]:
    """
    mode='hybrid'   both retrievers, fused with RRF (production path)
    mode='vector'   dense retrieval only      -- ablation
    mode='keyword'  BM25 only                 -- ablation
    """
    if mode not in {"hybrid", "vector", "keyword"}:
        raise ValueError(f"Unknown search mode: {mode!r}")

    vector_results = []
    keyword_results = []

    if mode in {"hybrid", "vector"}:
        vector_results = vector_db.search(embedder.embed_query(query), top_k=top_k)
    if mode in {"hybrid", "keyword"}:
        keyword_results = keyword_index.search(query, top_k=top_k)

    return reciprocal_rank_fusion(vector_results, keyword_results, k=rrf_k, top_k=top_k)
