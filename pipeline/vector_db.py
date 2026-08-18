"""
pipeline/vector_db.py
======================
FLOWCHART BLOCK: "Vector DB"

Responsibility: store chunk embeddings and answer "give me the top-k chunks
whose vectors are closest to this query vector".

We use FAISS's IndexFlatIP (exact inner-product search). With L2-normalised
vectors, inner product *is* cosine similarity, so no separate normalisation
step is needed at query time. Flat search is O(n) per query, which is the
right default at this corpus size: exact results, no recall loss, and no
approximation parameters to mistune. Approximate indexes only start paying for
themselves in the hundreds-of-thousands-of-vectors range.

Persistence is handled here rather than in the orchestrator, because how an
index serialises itself is the index's own business -- a Qdrant-backed
implementation of this same `add`/`search`/`save`/`load` interface would
persist server-side and make these methods no-ops.

Scaling path:
  - IndexFlatIP -> IndexIVFFlat / IndexHNSWFlat past ~1M vectors.
  - FAISS -> a managed vector DB (Qdrant, Weaviate, pgvector) once you need
    metadata filtering, multi-tenancy, or horizontal scale. The interface
    below is deliberately the shape you'd expose from such a client wrapper,
    so orchestrator.py would not change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np

from pipeline.parse_chunk import Chunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "vectors.faiss"
_CHUNKS_FILE = "vector_chunks.json"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner product == cosine sim on normalised vectors
        self._chunks: List[Chunk] = []        # parallel array: row i in the index <-> _chunks[i]

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> int:
        """
        Index the given chunks, replacing any document already present.

        Ingestion is idempotent at the *document* level, not the chunk level.
        Blindly appending would double the corpus every time a directory is
        re-ingested, and retrieval would then return the same passage at ranks
        one and two, spending the context budget on a single fact.

        Skipping chunks whose id already exists is the tempting shortcut and is
        worse: chunk ids derive from the file path, so an edited document
        produces the same ids as before and its changes would be silently
        discarded. Replacing by doc_id handles both cases -- re-ingesting
        unchanged files is a no-op, and re-ingesting edited files picks up the
        edit.

        Returns the number of chunks added.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"chunks/embeddings length mismatch: {len(chunks)} vs {embeddings.shape[0]}")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"expected embedding dim {self.dim}, got {embeddings.shape[1]}")
        if not chunks:
            return 0

        incoming_docs = {c.doc_id for c in chunks}
        superseded = self._retain_rows(
            [i for i, c in enumerate(self._chunks) if c.doc_id not in incoming_docs]
        )
        if superseded:
            logger.info("Replaced %d superseded chunks from %d document(s)", superseded, len(incoming_docs))

        self.index.add(embeddings.astype("float32"))
        self._chunks.extend(chunks)
        return len(chunks)

    def remove_documents(self, doc_ids: Set[str]) -> int:
        """
        Drop every chunk belonging to the given documents.

        Needed because a document deleted from the corpus must also disappear
        from the index. Nothing in an incoming batch can signal that a file has
        vanished, so removal has to be driven separately by the caller, which
        is the only party that knows what the corpus currently contains.

        Returns the number of chunks removed.
        """
        if not doc_ids:
            return 0
        return self._retain_rows([i for i, c in enumerate(self._chunks) if c.doc_id not in doc_ids])

    def document_sources(self) -> Dict[str, str]:
        """Map of doc_id -> source path for everything currently indexed."""
        return {c.doc_id: c.source for c in self._chunks}

    def _retain_rows(self, retained_rows: List[int]) -> int:
        """
        Rebuild the index keeping only the given rows, returning how many were
        dropped.

        A rebuild rather than an in-place delete because IndexFlat.remove_ids
        swaps the last vector into the freed slot. That would break the
        invariant this class depends on -- row i in the FAISS index is
        self._chunks[i] -- and every subsequent search would return the wrong
        text for the right vector, silently.
        """
        dropped = len(self._chunks) - len(retained_rows)
        if not dropped:
            return 0

        stored = self.index.reconstruct_n(0, self.index.ntotal)
        self.index = faiss.IndexFlatIP(self.dim)
        if retained_rows:
            self.index.add(stored[retained_rows].astype("float32"))
        self._chunks = [self._chunks[i] for i in retained_rows]

        return dropped

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[ScoredChunk]:
        if not self._chunks:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_embedding, min(top_k, len(self._chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than top_k results exist
                continue
            results.append(ScoredChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    # ------------------------------------------------------------------ #
    # Persistence: ingestion is expensive, restarts should not repeat it
    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / _INDEX_FILE))
        # Chunks are stored as JSON rather than pickled: it stays readable and
        # diffable, and loading an index never executes arbitrary code.
        (path / _CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str) -> "VectorDB":
        path = Path(directory)
        index = faiss.read_index(str(path / _INDEX_FILE))
        chunks = [Chunk(**c) for c in json.loads((path / _CHUNKS_FILE).read_text(encoding="utf-8"))]

        db = cls(dim=index.d)
        db.index = index
        db._chunks = chunks
        return db

    @staticmethod
    def exists(directory: str) -> bool:
        path = Path(directory)
        return (path / _INDEX_FILE).exists() and (path / _CHUNKS_FILE).exists()

    def __len__(self) -> int:
        return len(self._chunks)
