"""
evaluation/run_eval.py
=======================
Measures what the retrieval stack is actually worth.

Every RAG tutorial asserts that hybrid search beats single-retriever search
and that reranking helps. This harness checks whether that is true *on this
corpus*, by running four configurations against the same labelled question set
and reporting the difference:

    bm25            lexical retrieval only
    vector          dense retrieval only
    hybrid          both, fused with reciprocal rank fusion
    hybrid+rerank   fusion followed by cross-encoder reranking (production path)

Results are also broken down by query type. That breakdown is the interesting
part: hybrid search is supposed to win because BM25 rescues rare exact terms
that embeddings miss while dense retrieval rescues paraphrases that BM25
misses. If that story is true, it shows up as each single retriever winning
its own category and losing the other. If hybrid merely tracks whichever
retriever is stronger overall, the fusion is not earning its cost.

No LLM is involved, so this runs offline, for free, deterministically, and in
CI -- which is what makes it usable as a regression gate rather than a
one-off demo.

    python -m evaluation.run_eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from config import settings
from evaluation.metrics import aggregate, evaluate_one
from pipeline.orchestrator import RAGPipeline

GOLDEN_SET_PATH = Path("data/eval/golden_set.json")
RESULTS_PATH = Path("data/eval/results.json")
REPORT_PATH = Path("data/eval/RESULTS.md")

# (label, retrieval mode, use reranker)
CONFIGURATIONS = [
    ("bm25", "keyword", False),
    ("vector", "vector", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at {path}. Run from the repository root.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_configuration(
    pipeline: RAGPipeline,
    questions: List[dict],
    mode: str,
    use_rerank: bool,
    k: int,
) -> dict:
    per_question = []
    latencies = []

    for item in questions:
        started = time.perf_counter()
        results = pipeline.retrieve(item["question"], mode=mode, use_rerank=use_rerank, top_k=k)
        latencies.append((time.perf_counter() - started) * 1000)

        # Metrics are computed over source documents, not chunks -- see
        # evaluation/metrics.py for why that distinction matters.
        ranked_sources = [r.chunk.metadata.get("filename", r.chunk.source) for r in results]
        scores = evaluate_one(ranked_sources, item["relevant_docs"], k=k)

        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "query_type": item.get("query_type", "mixed"),
            "expected": item["relevant_docs"],
            "retrieved": ranked_sources,
            **scores,
        })

    by_type: Dict[str, dict] = {}
    for query_type in sorted({q["query_type"] for q in per_question}):
        subset = [
            {key: value for key, value in q.items() if isinstance(value, float)}
            for q in per_question if q["query_type"] == query_type
        ]
        by_type[query_type] = aggregate(subset)

    numeric = [{key: value for key, value in q.items() if isinstance(value, float)} for q in per_question]

    return {
        "overall": aggregate(numeric),
        "by_query_type": by_type,
        "latency_ms_avg": round(sum(latencies) / len(latencies), 1),
        "per_question": per_question,
    }


def _table(rows: List[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(results: dict, k: int, corpus_size: int, question_count: int) -> str:
    hit_key, ndcg_key = f"hit@{k}", f"ndcg@{k}"

    main_rows = [
        [
            f"**{label}**" if label == "hybrid+rerank" else label,
            f"{data['overall']['hit@1']:.3f}",
            f"{data['overall']['hit@3']:.3f}",
            f"{data['overall'][hit_key]:.3f}",
            f"{data['overall']['mrr']:.3f}",
            f"{data['overall'][ndcg_key]:.3f}",
            f"{data['latency_ms_avg']:.0f} ms",
        ]
        for label, data in results.items()
    ]

    query_types = sorted(next(iter(results.values()))["by_query_type"].keys())
    breakdown_rows = [
        [label] + [f"{data['by_query_type'][qt]['mrr']:.3f}" for qt in query_types]
        for label, data in results.items()
    ]

    dense = results["vector"]["overall"]
    fused = results["hybrid"]["overall"]
    best = results["hybrid+rerank"]["overall"]

    def pct(new: float, old: float) -> str:
        return f"{((new - old) / old * 100):+.1f}%" if old else "n/a"

    latency_delta = results["hybrid+rerank"]["latency_ms_avg"] - results["hybrid"]["latency_ms_avg"]

    return f"""# Retrieval Evaluation Results

Generated by `python -m evaluation.run_eval` -- regenerated on every run.
Do not edit by hand; record interpretation in `evaluation/ANALYSIS.md` instead.

Corpus: {corpus_size} chunks from `data/sample_docs`.
Questions: {question_count} labelled queries from `data/eval/golden_set.json`.
Metrics are computed over source documents rather than chunks, so they stay
comparable across chunking configurations. No language model is involved --
these measure retrieval only, the stage that caps everything downstream.

## Overall

{_table(main_rows, ["Configuration", "Hit@1", "Hit@3", f"Hit@{k}", "MRR", f"nDCG@{k}", "Avg latency"])}

## MRR by query type

{_table(breakdown_rows, ["Configuration"] + query_types)}

`lexical` questions hinge on a rare exact term; `semantic` questions are
paraphrases sharing few terms with the source; `mixed` sits between.

## Deltas (computed)

- Fusion vs dense, MRR: {pct(fused['mrr'], dense['mrr'])}
- Rerank vs fusion, MRR: {pct(best['mrr'], fused['mrr'])}
- Rerank vs fusion, Hit@1: {pct(best['hit@1'], fused['hit@1'])} (for +{latency_delta:.0f} ms/query)

Interpretation of these numbers lives in `evaluation/ANALYSIS.md`, one entry per
iteration.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against the golden set.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for hit@k and nDCG@k (default: 5)")
    parser.add_argument("--corpus", default=None, help="Override the corpus directory")
    args = parser.parse_args()

    golden = load_golden_set()
    questions = golden["questions"]

    pipeline = RAGPipeline()
    # Always rebuild from source so results reflect the current corpus and
    # chunking settings rather than a stale persisted index.
    corpus_size = pipeline.ingest(args.corpus or golden.get("corpus", settings.auto_ingest_dir), persist=False)

    print(f"\nCorpus: {corpus_size} chunks | Questions: {len(questions)} | k={args.k}\n")

    results = {}
    for label, mode, use_rerank in CONFIGURATIONS:
        print(f"  running {label:<14} ...", end="", flush=True)
        results[label] = run_configuration(pipeline, questions, mode, use_rerank, args.k)
        overall = results[label]["overall"]
        print(
            f" hit@{args.k}={overall[f'hit@{args.k}']:.3f}"
            f"  mrr={overall['mrr']:.3f}"
            f"  ndcg@{args.k}={overall[f'ndcg@{args.k}']:.3f}"
            f"  ({results[label]['latency_ms_avg']:.0f} ms/query)"
        )

    report = render_report(results, args.k, corpus_size, len(questions))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\n{report}")
    print(f"Wrote {REPORT_PATH} and {RESULTS_PATH}")


if __name__ == "__main__":
    main()
