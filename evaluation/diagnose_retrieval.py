"""
evaluation/diagnose_retrieval.py
================================
A debugging tool, not a metric report.

For every golden question it answers two *separate* questions:

  1. Is a relevant document in the CANDIDATE POOL -- the top `hybrid_top_k`
     results from hybrid search, BEFORE the reranker runs?
  2. Where does that document end up AFTER reranking?

Why split them? Because a Hit@1 miss has three very different causes, and they
need opposite fixes:

  - "not in pool"           -> retrieval never surfaced the doc. Fix = a bigger
                               candidate pool (hybrid_top_k) or a better embedder.
  - "in pool, reranked out" -> retrieval found it, the reranker dropped it out of
                               the final top-k. Fix = a better reranker.
  - "in pool, not ranked 1" -> reranker kept it but not first. Fix = a better
                               reranker.

The headline number is POOL RECALL: the fraction of questions whose answer is in
the pool at all. That is the *ceiling* a perfect reranker could reach -- so:

  pool recall close to Hit@1  -> the pool is fine, the reranker is the bottleneck
                                 (growing the pool will NOT help).
  pool recall much > Hit@1    -> the reranker is leaving points on the table.
  pool recall itself is low   -> retrieval is the bottleneck; grow the pool or
                                 improve the embedding model.

Run from the repo root:  python -m evaluation.diagnose_retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Set

from config import settings
from evaluation.metrics import to_document_ranking
from pipeline.orchestrator import RAGPipeline
from pipeline.query_rewrite import rule_based_rewrite

GOLDEN_SET_PATH = Path("data/eval/golden_set.json")


def _doc_ranking(reranked) -> List[str]:
    """Collapse a ranked chunk list to a ranked list of document filenames."""
    sources = [r.chunk.metadata.get("filename", r.chunk.source) for r in reranked]
    return to_document_ranking(sources)


def _best_rank(ranking: Sequence[str], relevant: Set[str]) -> Optional[int]:
    """1-indexed position of the first relevant document, or None if absent."""
    for position, source in enumerate(ranking, start=1):
        if source in relevant:
            return position
    return None


def _deep_ranks(pipeline: RAGPipeline, question: str, relevant: Set[str]):
    """
    Document-level rank of the relevant doc across the WHOLE corpus, for each
    retriever separately. Bypasses hybrid_top_k by querying the retrievers
    directly, so we can see a rank far below the pool cutoff.

    Returns (vector_rank, bm25_rank), each 1-indexed or None if the retriever
    never surfaces the document at all.
    """
    rewritten = rule_based_rewrite(question)  # retrieval uses the rewritten query
    depth = len(pipeline.vector_db)           # search the entire corpus

    query_vec = pipeline.embedder.embed_query(rewritten)
    vector_docs = to_document_ranking(
        [sc.chunk.metadata.get("filename", sc.chunk.source)
         for sc in pipeline.vector_db.search(query_vec, top_k=depth)]
    )
    bm25_docs = to_document_ranking(
        [sc.chunk.metadata.get("filename", sc.chunk.source)
         for sc in pipeline.keyword_index.search(rewritten, top_k=depth)]
    )
    return _best_rank(vector_docs, relevant), _best_rank(bm25_docs, relevant)


def main() -> None:
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    questions = golden["questions"]

    pipeline = RAGPipeline()
    corpus = golden.get("corpus", settings.auto_ingest_dir)
    n_chunks = pipeline.ingest(corpus, persist=False)

    pool_k = settings.hybrid_top_k     # size of the candidate pool
    final_k = settings.rerank_top_k    # what survives reranking

    print(f"\nCorpus: {n_chunks} chunks | Questions: {len(questions)}")
    print(f"Candidate pool size (hybrid_top_k) = {pool_k} | final top-k (rerank_top_k) = {final_k}\n")

    in_pool = 0
    hit1 = 0
    not_in_pool: List[dict] = []
    reranked_out: List[dict] = []
    in_pool_not_first: List[dict] = []

    for item in questions:
        question = item["question"]
        relevant = set(item["relevant_docs"])

        # The candidate pool: hybrid search, no reranker, full pool size.
        pool = _doc_ranking(
            pipeline.retrieve(question, mode="hybrid", use_rerank=False, top_k=pool_k)
        )
        # The production ranking: hybrid search + reranker, final top-k.
        final = _doc_ranking(
            pipeline.retrieve(question, mode="hybrid", use_rerank=True, top_k=final_k)
        )

        pool_rank = _best_rank(pool, relevant)      # None if not in pool
        final_rank = _best_rank(final, relevant)    # None if not in final top-k

        if pool_rank is not None:
            in_pool += 1
        if final_rank == 1:
            hit1 += 1
            continue  # a Hit@1 success -- nothing to diagnose

        record = {
            "id": item["id"], "pool_rank": pool_rank, "final_rank": final_rank,
            "question": question, "relevant": relevant,
        }
        if pool_rank is None:
            not_in_pool.append(record)
        elif final_rank is None:
            reranked_out.append(record)
        else:
            in_pool_not_first.append(record)

    total = len(questions)
    print("=" * 60)
    print(f"POOL RECALL@{pool_k}: {in_pool}/{total} = {in_pool / total:.3f}   "
          f"(ceiling for a perfect reranker)")
    print(f"Hit@1 (final):    {hit1}/{total} = {hit1 / total:.3f}")
    print(f"Gap left by the reranker: {(in_pool - hit1) / total:.3f}")
    print("=" * 60)

    def show(title: str, rows: List[dict]) -> None:
        print(f"\n{title}: {len(rows)}")
        for r in rows:
            pr = r["pool_rank"] if r["pool_rank"] is not None else "absent"
            fr = r["final_rank"] if r["final_rank"] is not None else "absent"
            print(f"  {r['id']}: pool_rank={pr}  final_rank={fr}")

    show(f"MISSES: not in pool  -> grow pool / better embeddings", not_in_pool)
    show(f"MISSES: in pool but reranked out of top-{final_k}  -> better reranker", reranked_out)
    show(f"MISSES: in pool, kept, but not ranked #1  -> better reranker", in_pool_not_first)

    # Deep dive on the "not in pool" cases: where does the answer actually rank,
    # across the whole corpus, for each retriever?
    if not_in_pool:
        total_docs = len({src for src in pipeline.vector_db.document_sources().values()})
        print("\n" + "=" * 60)
        print(f"TRUE RANK of the 'not in pool' misses (out of {total_docs} documents)")
        print("=" * 60)
        for r in not_in_pool:
            vec_rank, bm25_rank = _deep_ranks(pipeline, r["question"], r["relevant"])
            vr = vec_rank if vec_rank is not None else "never"
            br = bm25_rank if bm25_rank is not None else "never"
            near = pool_k + 10
            if vec_rank is not None and vec_rank <= near:
                verdict = f"just past the pool -> a modest hybrid_top_k bump likely catches it"
            else:
                verdict = f"buried deep -> embedding problem, a bigger pool will NOT help"
            print(f"  {r['id']}: vector_rank={vr}  bm25_rank={br}  ->  {verdict}")


if __name__ == "__main__":
    main()
