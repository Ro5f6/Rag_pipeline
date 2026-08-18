"""
tests/test_metrics.py
======================
Tests for the evaluation metrics.

Worth testing carefully: an evaluation harness with a subtly wrong metric is
worse than no harness at all, because it produces confident numbers that
justify the wrong decisions.
"""

import math

import pytest

from evaluation.metrics import (
    aggregate,
    evaluate_one,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
    to_document_ranking,
)


# --------------------------------------------------------------------- #
# Chunk ranking -> document ranking
# --------------------------------------------------------------------- #
def test_document_ranking_dedupes_and_keeps_the_best_position():
    assert to_document_ranking(["a.txt", "a.txt", "b.txt", "a.txt", "c.txt"]) == ["a.txt", "b.txt", "c.txt"]


def test_document_ranking_of_an_empty_result_is_empty():
    assert to_document_ranking([]) == []


# --------------------------------------------------------------------- #
# hit@k
# --------------------------------------------------------------------- #
def test_hit_at_k_respects_the_cutoff():
    ranking = ["x.txt", "y.txt", "z.txt", "target.txt"]

    assert hit_at_k(ranking, ["target.txt"], k=3) == 0.0
    assert hit_at_k(ranking, ["target.txt"], k=4) == 1.0


def test_hit_at_k_is_satisfied_by_any_relevant_document():
    assert hit_at_k(["b.txt"], ["a.txt", "b.txt"], k=1) == 1.0


def test_hit_at_k_with_no_results_is_zero():
    assert hit_at_k([], ["a.txt"], k=5) == 0.0


# --------------------------------------------------------------------- #
# MRR
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("position,expected", [(0, 1.0), (1, 0.5), (2, 1 / 3), (3, 0.25)])
def test_reciprocal_rank_is_one_over_the_first_hit_position(position, expected):
    ranking = ["filler.txt"] * 5
    ranking[position] = "target.txt"

    assert reciprocal_rank(ranking, ["target.txt"]) == pytest.approx(expected)


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_retrieved():
    assert reciprocal_rank(["a.txt", "b.txt"], ["c.txt"]) == 0.0


def test_reciprocal_rank_uses_the_first_hit_not_the_last():
    assert reciprocal_rank(["a.txt", "b.txt"], ["a.txt", "b.txt"]) == 1.0


# --------------------------------------------------------------------- #
# nDCG
# --------------------------------------------------------------------- #
def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a.txt", "b.txt", "c.txt"], ["a.txt", "b.txt"], k=3) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(["x.txt", "y.txt"], ["a.txt"], k=2) == 0.0


def test_ndcg_penalises_a_relevant_document_ranked_lower():
    good = ndcg_at_k(["a.txt", "x.txt", "y.txt"], ["a.txt"], k=3)
    bad = ndcg_at_k(["x.txt", "y.txt", "a.txt"], ["a.txt"], k=3)

    assert good == pytest.approx(1.0)
    assert bad < good
    assert bad == pytest.approx(1 / math.log2(4))


def test_ndcg_ignores_documents_beyond_the_cutoff():
    assert ndcg_at_k(["x.txt", "y.txt", "a.txt"], ["a.txt"], k=2) == 0.0


# --------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------- #
def test_evaluate_one_reports_every_metric():
    scores = evaluate_one(["a.txt", "a.txt", "b.txt"], ["b.txt"], k=5)

    assert set(scores) == {"hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"}
    # b.txt is second once chunks collapse to documents.
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["mrr"] == pytest.approx(0.5)


def test_evaluate_one_is_unaffected_by_repeated_chunks_from_one_document():
    """Scores must not inflate merely because a document was split more finely."""
    coarse = evaluate_one(["a.txt", "b.txt"], ["a.txt"], k=5)
    fine = evaluate_one(["a.txt", "a.txt", "a.txt", "b.txt"], ["a.txt"], k=5)

    assert coarse == fine


def test_aggregate_averages_each_metric():
    assert aggregate([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}
