"""
tests/test_eval_generation.py
=============================
Tests the deterministic parts of the generation-eval harness -- report
rendering, aggregation helpers, and the answer cache -- without ever calling a
judge model or the network. The RAGAS scoring itself is a gated experiment run
manually via `make eval-gen`, not in CI.
"""

import math

from evaluation.gen_cache import AnswerCache
from evaluation.eval_generation import _mean, render_report


def test_answer_cache_round_trips(tmp_path):
    cache = AnswerCache(path=tmp_path / "c.json")
    assert cache.get("groq", "qwen", "q1", "hybrid", True, 5) is None

    cache.put("groq", "qwen", "q1", "hybrid", True, 5, "answer [1]", ["ctx a", "ctx b"])
    cache.save()

    reloaded = AnswerCache(path=tmp_path / "c.json")
    hit = reloaded.get("groq", "qwen", "q1", "hybrid", True, 5)
    assert hit["answer"] == "answer [1]"
    assert hit["contexts"] == ["ctx a", "ctx b"]


def test_cache_key_is_sensitive_to_generator_and_retrieval(tmp_path):
    cache = AnswerCache(path=tmp_path / "c.json")
    cache.put("groq", "qwen", "q1", "hybrid", True, 5, "a", [])
    # A different model, mode, or top_k must miss.
    assert cache.get("groq", "gpt-oss", "q1", "hybrid", True, 5) is None
    assert cache.get("groq", "qwen", "q1", "vector", True, 5) is None
    assert cache.get("groq", "qwen", "q1", "hybrid", True, 8) is None


def test_disabled_cache_never_stores(tmp_path):
    cache = AnswerCache(path=tmp_path / "c.json", enabled=False)
    cache.put("groq", "qwen", "q1", "hybrid", True, 5, "a", [])
    assert cache.get("groq", "qwen", "q1", "hybrid", True, 5) is None


def test_mean_ignores_none_and_nan():
    assert _mean([0.8, 1.0, 0.6]) == 0.8
    assert _mean([1.0, None, 0.0]) == 0.5           # None dropped
    assert math.isnan(_mean([None, float("nan")]))  # nothing survives -> NaN
    assert math.isnan(_mean([]))


def test_render_report_names_generator_judge_and_metrics():
    metric_names = ["faithfulness", "answer_relevancy", "context_precision_without_reference"]
    overall = {
        "faithfulness": 0.94,
        "answer_relevancy": 0.88,
        "context_precision_without_reference": 0.81,
        "cited_fraction": 0.72,
    }
    by_type = {
        "lexical": {m: 0.9 for m in metric_names},
        "semantic": {m: 0.7 for m in metric_names},
    }

    report = render_report(overall, by_type, metric_names, "qwen/qwen3.6-27b (groq)",
                           "gemini-2.5-flash (google_genai)", n=42)

    # Provenance is recorded so numbers aren't compared across judges.
    assert "qwen/qwen3.6-27b (groq)" in report
    assert "gemini-2.5-flash (google_genai)" in report
    assert "Questions scored: **42**" in report
    # Pretty metric labels and the citation stat are present.
    assert "Faithfulness" in report and "Context precision" in report
    assert "Answers with a citation" in report
    # By-query-type rows render.
    assert "lexical" in report and "semantic" in report
