"""
pipeline/parse_chunk.py
========================
FLOWCHART BLOCK: "Parse and chunk"

Responsibility: turn raw Document text into small, retrievable Chunk objects.

Why chunk at all?
- Embedding models and LLM context windows have finite size; an embedding of a
  200-page PDF is an average of everything it discusses, which places it near
  nothing in particular in vector space and matches nothing precisely.
- Smaller chunks retrieve more precisely -- a single paragraph about
  PagedAttention scores higher against a PagedAttention query than a whole
  document does.

Why overlap?
- Without it, a hard cut can split a definition from its explanation, leaving
  neither half retrievable on its own. Overlap keeps a sliding window of
  continuity so no idea is severed at a boundary.

Why recursive splitting, and only recursive splitting?
- RecursiveCharacterTextSplitter tries a prioritised list of separators --
  paragraph breaks, then line breaks, then sentences, then spaces -- and only
  falls back to a hard character cut when no natural boundary fits the size
  budget. It respects document structure far more often than a fixed-width
  window, which produces fewer chunks that begin or end mid-thought.
- There is deliberately no secondary strategy to fall back to. A fallback that
  silently switches chunking behaviour when something goes wrong means the
  corpus can be split two different ways depending on a transient failure, and
  the retrieval metrics in evaluation/ would then describe a configuration
  that is not necessarily the one running. Chunking is cheap and deterministic;
  if it fails, the right response is to fail loudly at ingest rather than to
  quietly index a differently-shaped corpus.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline.documents import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_and_chunk(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """
    Split documents into overlapping chunks.

    Raises ValueError if chunk_overlap is not smaller than chunk_size -- the
    splitter validates this on construction, before any document is read, so
    a misconfiguration fails immediately rather than partway through a corpus.
    """
    # Built once rather than per document: the separator list and its
    # configuration are identical for every document in the batch.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[Chunk] = []

    for doc in documents:
        text = doc.text.strip()
        if not text:
            continue

        # Filtering before enumerate keeps chunk_index contiguous, which
        # matters because chunk ids are built from it and are user-visible in
        # citations.
        split_texts = [t for t in splitter.split_text(text) if t.strip()]

        for chunk_idx, chunk_text in enumerate(split_texts):
            chunks.append(
                Chunk(
                    id=f"{doc.id}_{chunk_idx}",
                    doc_id=doc.id,
                    text=chunk_text,
                    source=doc.source,
                    metadata={**doc.metadata, "chunk_index": chunk_idx},
                )
            )

    return chunks
