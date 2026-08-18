"""
pipeline/embed.py
==================
FLOWCHART BLOCK: "Embed"

Responsibility: turn chunk text into dense vectors that capture semantic
meaning, so that "near" vectors mean "similar meaning" -- this is what makes
semantic search possible (as opposed to keyword matching).

We use sentence-transformers/all-MiniLM-L6-v2: small (80MB), fast on CPU,
384-dim output. Good for learning/prototyping. In production you'd likely
swap to a larger model (e.g. bge-large, e5-large, or a hosted embeddings API)
-- again, a drop-in replacement behind the same `embed_texts` interface.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        """
        The model's true output dimension.

        Downstream (vector_db.py) sizes its index from this rather than from a
        configured constant, so changing embedding_model can never leave the
        index expecting a width the encoder no longer produces.
        """
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns a (len(texts), dim) float32 array, L2-normalized.
        Normalizing here means downstream we can use a plain dot product
        as cosine similarity inside FAISS (IndexFlatIP), which is cheaper
        than computing cosine similarity from scratch every query.
        """
        if not texts:
            return np.empty((0, self.dim), dtype="float32")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
