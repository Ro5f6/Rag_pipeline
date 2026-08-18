"""
pipeline/orchestrator.py
==========================
This module IS the flowchart, expressed as code:

    Documents -> Parse and chunk -> Embed -> Vector DB \\
                                  -> Keyword index      -> Hybrid search
    User query -> Query rewrite  --------------------- /
                                                            |
                                                            v
                                                          Rerank
                                                            |
                                                            v
                                                       Context format
                                                            |
                                                            v
                                                        LLM generate
                                                            |
                                                            v
                                                       Cite sources
                                                            |
                                                            v
                                                          User

Two entry points mirror the two paths through the diagram:

  ingest()   the left-hand side. Runs when the knowledge base changes.
             Expensive (embedding every chunk), batch-shaped, and idempotent.
  query()    the right-hand side. Runs on every user request. Latency-critical.

Keeping them separate is what allows them to scale independently: ingestion
belongs in a background worker or a scheduled job, while query sits behind an
autoscaled API. Fusing them into one function -- the tempting shortcut -- is
what forces a re-embed of the entire corpus on every deploy.

Components are built lazily and reused across requests. Loading the embedding
and reranking models takes seconds; doing that per request would dominate
latency, and doing it at import time would make the module impossible to
import in tests or in a CI job that only wants to check chunking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import Settings
from config import settings as default_settings
from pipeline.cite_sources import Citation, attach_citations
from pipeline.context_format import build_prompt, format_context
from pipeline.documents import Document, corpus_fingerprint, load_documents_from_dir
from pipeline.embed import Embedder
from pipeline.hybrid_search import hybrid_search
from pipeline.keyword_index import KeywordIndex
from pipeline.llm_generate import Generator, build_generator
from pipeline.observability import Stopwatch, configure_logging
from pipeline.parse_chunk import Chunk, parse_and_chunk
from pipeline.query_rewrite import rule_based_rewrite
from pipeline.rerank import RerankedChunk, Reranker
from pipeline.vector_db import VectorDB

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "manifest.json"


@dataclass
class RAGResponse:
    """Everything the caller needs to render, audit, and debug one answer."""

    answer: str
    citations: List[Citation]
    backend: str                              # which generator answered
    model: str
    retrieval_mode: str
    rewritten_query: str
    candidates_retrieved: int                 # before reranking
    chunks_used: int                          # after reranking
    timings_ms: Dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, config: Optional[Settings] = None):
        self.config = config or default_settings
        configure_logging(self.config.log_level)

        # Lazily constructed -- see the properties below.
        self._embedder: Optional[Embedder] = None
        self._vector_db: Optional[VectorDB] = None
        self._keyword_index: Optional[KeywordIndex] = None
        self._reranker: Optional[Reranker] = None
        self._generator: Optional[Generator] = None

    # ---------------------------------------------------------------- #
    # Lazily built, then cached for the process lifetime
    # ---------------------------------------------------------------- #
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            logger.info("Loading embedding model: %s", self.config.embedding_model)
            self._embedder = Embedder(self.config.embedding_model)
        return self._embedder

    @property
    def vector_db(self) -> VectorDB:
        if self._vector_db is None:
            # The index dimension comes from the embedder itself, never from a
            # hardcoded constant, so swapping embedding_model cannot silently
            # desync the index shape from the vectors written into it.
            self._vector_db = VectorDB(dim=self.embedder.dim)
        return self._vector_db

    @property
    def keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex()
        return self._keyword_index

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            logger.info("Loading reranker model: %s", self.config.rerank_model)
            self._reranker = Reranker(self.config.rerank_model)
        return self._reranker

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = build_generator(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                timeout=self.config.llm_timeout_seconds,
            )
        return self._generator

    # ---------------------------------------------------------------- #
    # LEFT SIDE OF THE DIAGRAM: build the knowledge base
    # ---------------------------------------------------------------- #
    def ingest(self, directory: str, persist: Optional[bool] = None) -> int:
        """Documents -> Parse and chunk -> Embed -> Vector DB + Keyword index"""
        persist = self.config.persist_indexes if persist is None else persist

        documents: List[Document] = load_documents_from_dir(directory)
        if not documents:
            raise FileNotFoundError(f"No documents found in {directory!r}")

        chunks: List[Chunk] = parse_and_chunk(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        self._drop_deleted_documents(directory, present={d.id for d in documents})

        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        added = self.vector_db.add(chunks, embeddings)
        self.keyword_index.add(chunks)

        logger.info(
            "Ingested %d documents -> %d chunks (%d total indexed)",
            len(documents), added, len(self.vector_db),
        )

        if persist:
            self.save(directory)

        return len(chunks)

    def _drop_deleted_documents(self, directory: str, present: set) -> None:
        """
        Treat `directory` as authoritative for the documents it contains.

        A document removed from disk must also leave the index. Nothing in an
        incoming batch can signal a deletion -- the file simply isn't there to
        be loaded -- so it has to be detected by comparing what the directory
        holds now against what the index still remembers about it.

        Scoped by directory on purpose: re-ingesting ./docs/a must never evict
        documents that were ingested from ./docs/b.
        """
        scope = Path(directory).resolve()

        vanished = {
            doc_id
            for doc_id, source in self.vector_db.document_sources().items()
            if doc_id not in present and Path(source).resolve().parent == scope
        }
        if not vanished:
            return

        removed = self.vector_db.remove_documents(vanished)
        self.keyword_index.remove_documents(vanished)
        logger.info("Dropped %d chunks from %d document(s) deleted from %s", removed, len(vanished), directory)

    def save(self, source_directory: str) -> None:
        index_dir = Path(self.config.index_dir)
        self.vector_db.save(str(index_dir))
        self.keyword_index.save(str(index_dir))

        # The manifest records which corpus this index was built from, and with
        # which embedding model. Both matter: vectors from a different model are
        # not comparable, so reusing an index across a model change silently
        # corrupts every search result.
        (index_dir / _MANIFEST_FILE).write_text(
            json.dumps({
                "corpus_dir": source_directory,
                "corpus_fingerprint": corpus_fingerprint(source_directory),
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "chunks": len(self.vector_db),
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted %d chunks to %s", len(self.vector_db), index_dir)

    def _manifest_is_current(self, directory: str) -> bool:
        """Is the persisted index still valid for this corpus and configuration?"""
        manifest_path = Path(self.config.index_dir) / _MANIFEST_FILE
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        checks = {
            "corpus": manifest.get("corpus_fingerprint") == corpus_fingerprint(directory),
            "embedding model": manifest.get("embedding_model") == self.config.embedding_model,
            "chunk size": manifest.get("chunk_size") == self.config.chunk_size,
            "chunk overlap": manifest.get("chunk_overlap") == self.config.chunk_overlap,
        }

        stale = [name for name, ok in checks.items() if not ok]
        if stale:
            logger.info("Persisted index is stale (%s changed) -- rebuilding.", ", ".join(stale))
            return False
        return True

    def load(self) -> bool:
        """Restore persisted indexes. Returns False if none exist."""
        if not (VectorDB.exists(self.config.index_dir) and KeywordIndex.exists(self.config.index_dir)):
            return False

        self._vector_db = VectorDB.load(self.config.index_dir)
        self._keyword_index = KeywordIndex.load(self.config.index_dir)
        logger.info("Loaded %d chunks from %s", len(self._vector_db), self.config.index_dir)
        return True

    def load_or_ingest(self, directory: Optional[str] = None) -> int:
        """
        Startup path: reuse a persisted index when it is still valid for the
        current corpus and settings, otherwise rebuild it.

        Returns the total number of chunks available -- unlike ingest(), which
        returns how many were newly added. Callers here want "is the knowledge
        base ready and how big is it", not "how much work did that take".
        """
        directory = directory or self.config.auto_ingest_dir

        if not (self.config.persist_indexes and self._manifest_is_current(directory) and self.load()):
            self.ingest(directory)

        return len(self.vector_db)

    # ---------------------------------------------------------------- #
    # RIGHT SIDE OF THE DIAGRAM: answer a user question
    # ---------------------------------------------------------------- #
    def retrieve(
        self,
        user_query: str,
        mode: str = "hybrid",
        use_rerank: bool = True,
        top_k: Optional[int] = None,
    ) -> List[RerankedChunk]:
        """
        Retrieval only, with no generation.

        Split out from query() so the evaluation harness can measure retrieval
        quality directly -- the part of a RAG system that actually determines
        whether a correct answer is even possible.
        """
        rewritten = rule_based_rewrite(user_query)

        candidates = hybrid_search(
            rewritten,
            self.vector_db,
            self.keyword_index,
            self.embedder,
            top_k=self.config.hybrid_top_k,
            rrf_k=self.config.rrf_k,
            mode=mode,
        )

        final_k = top_k or self.config.rerank_top_k
        if use_rerank:
            # Deliberately the ORIGINAL query, not the rewritten one. Rewriting
            # exists to help lexical and dense *retrieval* -- it lowercases text
            # and appends acronym expansions, which is useful for matching and
            # actively harmful for a cross-encoder judging semantic relevance
            # against what the user actually asked.
            return self.reranker.rerank(user_query, candidates, top_k=final_k)

        # Ablation path: keep fusion order, skip the cross-encoder.
        return [RerankedChunk(chunk=c.chunk, rerank_score=c.rrf_score) for c in candidates[:final_k]]

    def query(
        self,
        user_query: str,
        mode: str = "hybrid",
        use_rerank: bool = True,
    ) -> RAGResponse:
        watch = Stopwatch()

        # User query -> Query rewrite
        with watch.stage("query_rewrite"):
            rewritten = rule_based_rewrite(user_query)

        # (Query rewrite, Vector DB, Keyword index) -> Hybrid search
        with watch.stage("retrieval"):
            candidates = hybrid_search(
                rewritten,
                self.vector_db,
                self.keyword_index,
                self.embedder,
                top_k=self.config.hybrid_top_k,
                rrf_k=self.config.rrf_k,
                mode=mode,
            )

        # Hybrid search -> Rerank
        with watch.stage("rerank"):
            if use_rerank:
                # Original query, not the rewritten one -- see retrieve().
                top_chunks = self.reranker.rerank(user_query, candidates, top_k=self.config.rerank_top_k)
            else:
                top_chunks = [
                    RerankedChunk(chunk=c.chunk, rerank_score=c.rrf_score)
                    for c in candidates[: self.config.rerank_top_k]
                ]

        # Rerank -> Context format
        with watch.stage("context_format"):
            context = format_context(top_chunks)
            prompt = build_prompt(user_query, context)

        # Context format -> LLM generate
        with watch.stage("generation"):
            raw_answer = self.generator.generate(prompt, query=user_query, chunks=top_chunks)

        # LLM generate -> Cite sources -> User
        with watch.stage("cite_sources"):
            final = attach_citations(raw_answer, top_chunks)

        return RAGResponse(
            answer=final.answer,
            citations=final.citations,
            backend=self.generator.name,
            model=self.generator.model,
            retrieval_mode=mode,
            rewritten_query=rewritten,
            candidates_retrieved=len(candidates),
            chunks_used=len(top_chunks),
            timings_ms=watch.as_dict(),
        )


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG pipeline from the command line.")
    parser.add_argument("--query", "-q", help="Question to answer")
    parser.add_argument("--dir", "-d", default=None, help="Corpus directory to ingest")
    parser.add_argument("--ingest-only", action="store_true", help="Build and persist indexes, then exit")
    parser.add_argument("--rebuild", action="store_true", help="Ignore any persisted index and re-ingest")
    parser.add_argument(
        "--mode", default="hybrid", choices=["hybrid", "vector", "keyword"],
        help="Retrieval strategy (default: hybrid)",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.rebuild or args.ingest_only:
        pipeline.ingest(args.dir or pipeline.config.auto_ingest_dir)
    else:
        pipeline.load_or_ingest(args.dir)
    print(f"Knowledge base ready: {len(pipeline.vector_db)} chunks.\n")

    if args.ingest_only:
        return

    question = args.query or "How does vLLM reduce GPU memory waste?"
    print(f"Q: {question}\n")

    result = pipeline.query(question, mode=args.mode, use_rerank=not args.no_rerank)

    print(result.answer)
    print("\nSources:")
    for c in result.citations:
        print(f"  {c.marker} {c.filename} (chunk {c.chunk_id})")

    print(f"\nBackend: {result.backend} ({result.model})")
    print("Timings (ms): " + ", ".join(f"{k}={v}" for k, v in result.timings_ms.items()))

    if result.backend == "extractive":
        print(
            "\nNote: answers are being composed extractively from retrieved chunks. "
            "Set an API key (see .env.example) for LLM-written answers."
        )


if __name__ == "__main__":
    _cli()
