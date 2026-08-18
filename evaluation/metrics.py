"""
evaluation/metrics.py
======================
Standard information-retrieval metrics, implemented directly rather than
imported, because the definitions are short and the details are where
evaluation harnesses quietly go wrong.

A NOTE ON THE UNIT OF EVALUATION
--------------------------------------------------------------------------
Retrieval returns *chunks*, but the golden set labels *documents*. Scoring
chunks directly would make the metrics depend on chunk size: a corpus split
into smaller pieces produces more relevant chunks per document and inflates
every score without retrieval having improved at all.

So the ranked chunk list is collapsed to a ranked document list, keeping each
document's best (earliest) position. Metrics are then computed over documents,
which keeps them comparable across chunking configurations -- exactly the
comparison this harness exists to support.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def to_document_ranking(ranked_sources: Sequence[str]) -> List[str]:
    """Collapse a ranked chunk list to a ranked document list, best rank wins."""
    seen: List[str] = []
    for source in ranked_sources:
        if source not in seen:
            seen.append(source)
    return seen


def hit_at_k(ranking: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if any relevant document appears in the top k, else 0.0.

    The bluntest and most important metric: it answers whether the information
    needed to answer the question reached the model's context at all. Every
    downstream generation metric is capped by this one.
    """
    return 1.0 if set(ranking[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranking: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant document (0.0 if none appear).

    Rewards putting the right document first rather than fifth, which matters
    because models attend unevenly across a context window.
    """
    relevant_set = set(relevant)
    for position, source in enumerate(ranking, start=1):
        if source in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Normalised discounted cumulative gain with binary relevance.

    Every relevant document contributes, discounted logarithmically by its
    position, then divided by the score a perfect ranking would achieve. Unlike
    hit rate it distinguishes "both relevant docs at ranks 1 and 2" from "one
    at rank 1 and one at rank 9".
    """
    relevant_set = set(relevant)

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, source in enumerate(ranking[:k], start=1)
        if source in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))

    return dcg / idcg if idcg else 0.0


def evaluate_one(ranked_sources: Sequence[str], relevant: Sequence[str], k: int = 5) -> Dict[str, float]:
    """
    All metrics for a single question.

    hit@1 is reported alongside hit@k deliberately. On a corpus of this size
    hit@5 saturates at 1.000 for every configuration, which makes it useless
    for comparing them -- a metric every candidate passes measures nothing.
    hit@1 asks the much harder question of whether the correct document was
    ranked first, and that is where configurations actually separate.
    """
    ranking = to_document_ranking(ranked_sources)
    return {
        "hit@1": hit_at_k(ranking, relevant, 1),
        "hit@3": hit_at_k(ranking, relevant, 3),
        f"hit@{k}": hit_at_k(ranking, relevant, k),
        "mrr": reciprocal_rank(ranking, relevant),
        f"ndcg@{k}": ndcg_at_k(ranking, relevant, k),
    }


def aggregate(per_question: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across questions."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {key: round(sum(q[key] for q in per_question) / len(per_question), 4) for key in keys}
