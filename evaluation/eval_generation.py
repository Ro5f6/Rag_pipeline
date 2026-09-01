"""
evaluation/eval_generation.py
=============================
Measures GENERATION quality with RAGAS -- the answer stage, not retrieval.

Where run_eval.py asks "did we retrieve the right document?", this asks "given
what we retrieved, is the written answer any good?". It runs the production
pipeline over the golden set, then scores each answer with three reference-free
RAGAS metrics (no gold answer required):

    faithfulness        are the answer's claims grounded in the retrieved context?
    answer_relevancy    does the answer actually address the question?
    context_precision   are the relevant chunks ranked near the top?

Unlike run_eval.py this is NOT deterministic and NOT free: every metric is an
LLM-judge call. So it is a gated experiment (its own `make eval-gen` target,
never CI), the judge is a different model from the generator (RAG_JUDGE_* vs
RAG_LLM_*), and generated answers are cached so re-scoring never regenerates.

    python -m evaluation.eval_generation --sample 20

Note: the generator is configured by RAG_LLM_* (this project ships Groq/qwen);
make sure no stale RAG_LLM_* shell env vars are shadowing your .env.
"""

from __future__ import annotations

import argparse
import json
import asyncio
import logging
from datetime import date
from pathlib import Path
from typing import List, Tuple

from config import settings
from evaluation.gen_cache import AnswerCache
from evaluation.ragas_judge import build_judge
from pipeline.orchestrator import RAGPipeline

# _ragas_compat is imported (and applied) inside ragas_judge before ragas loads,
# so the collections metrics below import cleanly.
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/eval/golden_set.json")
GEN_RESULTS_PATH = Path("data/eval/gen_results.json")
GEN_REPORT_PATH = Path("data/eval/GEN_RESULTS.md")

# Reference-free metrics only. Correctness / recall (which need a gold answer)
# arrive with the key-facts golden-set extension in a later iteration.
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision_without_reference"]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at {path}. Run from the repository root.")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_answers(pipeline: RAGPipeline, questions: List[dict], cache: AnswerCache) -> Tuple[List[dict], str, str]:
    """Run the pipeline over the questions, returning RAGAS-shaped rows.

    Cached answers are reused; only cache misses call the generator.
    """
    backend, model = pipeline.generator.name, pipeline.generator.model
    mode, rerank, top_k = "hybrid", True, settings.rerank_top_k
    rows: List[dict] = []

    for i, q in enumerate(questions, start=1):
        cached = cache.get(backend, model, q["id"], mode, rerank, top_k)
        if cached:
            answer, contexts = cached["answer"], cached["contexts"]
        else:
            try:
                resp = pipeline.query(q["question"])
            except Exception as exc:  # noqa: BLE001 - one bad call shouldn't kill the run
                logger.warning("Generation failed for %s (%s); skipping.", q["id"], exc)
                continue
            answer = resp.answer
            contexts = [c.chunk.text for c in resp.context_chunks]
            cache.put(backend, model, q["id"], mode, rerank, top_k, answer, contexts)

        rows.append({
            "id": q["id"],
            "user_input": q["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "query_type": q.get("query_type", "mixed"),
        })
        print(f"  generated {i}/{len(questions)}", end="\r", flush=True)

    cache.save()
    print()
    return rows, backend, model


def score(rows: List[dict], settings_) -> Tuple[dict, dict, List[dict], List[str]]:
    """Score the rows with the RAGAS collections metrics (async, concurrency-limited).

    Returns (overall, by_query_type, per_question, metric_names). A metric that
    fails on a row (rate limit, truncated judge output) is recorded as None
    rather than aborting the run.
    """
    judge, embeddings = build_judge(settings_)
    faithfulness = Faithfulness(llm=judge)
    answer_relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)
    context_precision = ContextPrecisionWithoutReference(llm=judge)

    async def score_row(row: dict, sem: asyncio.Semaphore) -> dict:
        result = {"id": row["id"], "query_type": row["query_type"]}
        async with sem:
            for name, metric, needs_ctx in (
                ("faithfulness", faithfulness, True),
                ("answer_relevancy", answer_relevancy, False),
                ("context_precision_without_reference", context_precision, True),
            ):
                try:
                    if needs_ctx:
                        r = await metric.ascore(user_input=row["user_input"], response=row["response"],
                                                retrieved_contexts=row["retrieved_contexts"])
                    else:
                        r = await metric.ascore(user_input=row["user_input"], response=row["response"])
                    result[name] = float(r.value) if r.value is not None else None
                except Exception as exc:  # noqa: BLE001 - one failed metric shouldn't kill the run
                    logger.warning("Metric %s failed for %s: %s", name, row["id"], exc)
                    result[name] = None
        return result

    async def run_all() -> List[dict]:
        sem = asyncio.Semaphore(settings_.eval_max_workers)
        return await asyncio.gather(*(score_row(r, sem) for r in rows))

    per_question = asyncio.run(run_all())

    overall = {name: _mean(q[name] for q in per_question) for name in METRIC_NAMES}
    overall["cited_fraction"] = round(sum("[1]" in r["response"] or "[2]" in r["response"] for r in rows) / len(rows), 3)

    by_type = {}
    for qt in sorted({q["query_type"] for q in per_question}):
        subset = [q for q in per_question if q["query_type"] == qt]
        by_type[qt] = {name: _mean(q[name] for q in subset) for name in METRIC_NAMES}

    return overall, by_type, per_question, METRIC_NAMES


def _mean(values) -> float:
    """Mean of the non-None, non-NaN values; NaN if none survive."""
    xs = [v for v in values if v is not None and v == v]
    return round(sum(xs) / len(xs), 3) if xs else float("nan")


def _table(rows: List[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


_PRETTY = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision_without_reference": "Context precision",
}


def render_report(overall, by_type, metric_names, generator, judge_desc, n) -> str:
    pretty = [_PRETTY.get(m, m) for m in metric_names]

    overall_rows = [[label, f"{overall[m]:.3f}"] for label, m in zip(pretty, metric_names)]
    overall_rows.append(["Answers with a citation", f"{overall['cited_fraction']:.3f}"])

    by_type_rows = [
        [qt] + [f"{by_type[qt][m]:.3f}" for m in metric_names]
        for qt in sorted(by_type)
    ]

    return f"""# Generation Evaluation Results (RAGAS, reference-free)

Generated by `python -m evaluation.eval_generation` -- regenerated on every run.
Do not edit by hand; record interpretation in `evaluation/ANALYSIS.md` instead.

Generator: **{generator}**  |  Judge: **{judge_desc}**
Questions scored: **{n}**  |  Retrieval: hybrid + rerank  |  Date: {date.today().isoformat()}

These are LLM-judged and non-deterministic; the judge model and date are recorded
above so a number is never compared across a different judge.

## Overall

{_table(overall_rows, ["Metric", "Score"])}

## By query type

{_table(by_type_rows, ["Query type"] + pretty)}

`faithfulness` = claims grounded in context; `answer relevancy` = on-topic-ness;
`context precision` = relevant chunks ranked high. "Answers with a citation" is a
deterministic check (does the answer contain a `[n]` marker), not a judged score.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation quality with RAGAS (reference-free).")
    parser.add_argument("--sample", type=int, default=None, help="Score only the first N questions (default: RAG_EVAL_SAMPLE_SIZE or all)")
    parser.add_argument("--corpus", default=None, help="Override the corpus directory")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached answers and regenerate")
    args = parser.parse_args()

    golden = load_golden_set()
    questions = golden["questions"]
    limit = args.sample if args.sample is not None else settings.eval_sample_size
    if limit and limit > 0:
        questions = questions[:limit]

    pipeline = RAGPipeline()
    pipeline.load_or_ingest(args.corpus or golden.get("corpus", settings.auto_ingest_dir))

    cache = AnswerCache(enabled=not args.no_cache)
    print(f"\nGenerating answers for {len(questions)} questions ...")
    rows, backend, model = generate_answers(pipeline, questions, cache)
    if not rows:
        raise RuntimeError("No answers were generated -- check the generator configuration (RAG_LLM_*).")

    print(f"Scoring {len(rows)} answers with the judge ({settings.judge_model}) ...")
    overall, by_type, per_question, metric_names = score(rows, settings)

    generator_desc = f"{model} ({backend})"
    judge_desc = f"{settings.judge_model} ({settings.judge_provider})"
    report = render_report(overall, by_type, metric_names, generator_desc, judge_desc, len(rows))

    GEN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEN_RESULTS_PATH.write_text(json.dumps({
        "generator": generator_desc,
        "judge": judge_desc,
        "n": len(rows),
        "date": date.today().isoformat(),
        "overall": overall,
        "by_query_type": by_type,
        "per_question": per_question,
    }, indent=2), encoding="utf-8")
    GEN_REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\n{report}")
    print(f"Wrote {GEN_REPORT_PATH} and {GEN_RESULTS_PATH}")


if __name__ == "__main__":
    main()
