"""
pipeline/rerank.py
====================
FLOWCHART BLOCK: "Rerank"

Responsibility: take the ~20 candidates hybrid search returned (fast, coarse
retrieval) and re-score the top handful with a slower but much more accurate
model, so only the best 3-5 chunks reach the LLM's context window.

Why not just retrieve fewer candidates in hybrid search directly? Because
embeddings and BM25 are "bi-encoders" -- they score the query and the chunk
*independently* then compare vectors, which is fast but loses cross-term
interaction. A cross-encoder reranker feeds the (query, chunk) pair together
through a transformer, letting it directly attend to how specific words in
the query relate to specific words in the chunk. Far more accurate, but too
slow to run over your entire corpus -- hence "retrieve broad and cheap, then
rerank narrow and precise."

This is precisely the two-stage retrieval pattern used in most production
search/RAG systems (and matches your "Inference optimization" / retrieval
phase reading).
"""

from dataclasses import dataclass
from typing import List
from sentence_transformers import CrossEncoder

from pipeline.hybrid_search import FusedChunk
from pipeline.parse_chunk import Chunk


@dataclass
class RerankedChunk:
    chunk: Chunk
    rerank_score: float


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[FusedChunk], top_k: int = 5) -> List[RerankedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)  # higher = more relevant

        reranked = [
            RerankedChunk(chunk=c.chunk, rerank_score=float(s))
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]
