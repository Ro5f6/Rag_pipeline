"""
pipeline/cite_sources.py
===========================
FLOWCHART BLOCK: "Cite sources"

Responsibility: attach the actual source metadata (filenames, chunk ids) to
the bracket citations ([1], [2]) the LLM emitted in its answer, so the
final response returned to the user is fully auditable -- you can trace
every claim back to a real chunk in your knowledge base.

This is what separates a trustworthy RAG answer from an opaque one: the LLM
never invents a source; it can only reference numbers you handed it, and
this block maps those numbers back to ground truth.
"""

from dataclasses import dataclass
from typing import List, Dict
from pipeline.rerank import RerankedChunk


@dataclass
class Citation:
    marker: str          # e.g. "[1]"
    filename: str
    chunk_id: str
    text_preview: str


@dataclass
class FinalAnswer:
    answer: str
    citations: List[Citation]


def build_citation_map(reranked_chunks: List[RerankedChunk]) -> Dict[str, Citation]:
    citation_map = {}
    for i, rc in enumerate(reranked_chunks, start=1):
        marker = f"[{i}]"
        citation_map[marker] = Citation(
            marker=marker,
            filename=rc.chunk.metadata.get("filename", rc.chunk.source),
            chunk_id=rc.chunk.id,
            text_preview=rc.chunk.text[:150],
        )
    return citation_map


def attach_citations(llm_answer: str, reranked_chunks: List[RerankedChunk]) -> FinalAnswer:
    citation_map = build_citation_map(reranked_chunks)

    # Only keep citations that were actually referenced in the answer text,
    # so the user doesn't see a "Sources" list padded with unused chunks.
    used = [c for marker, c in citation_map.items() if marker in llm_answer]

    return FinalAnswer(answer=llm_answer, citations=used)


def render_final_answer(final: FinalAnswer) -> str:
    lines = [final.answer, "", "Sources:"]
    for c in final.citations:
        lines.append(f"  {c.marker} {c.filename} (chunk {c.chunk_id})")
    return "\n".join(lines)
