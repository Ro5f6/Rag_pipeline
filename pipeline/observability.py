"""
pipeline/observability.py
==========================
Cross-cutting concern, not a flowchart block.

The single most useful thing to know about a RAG system in production is
*which stage* is slow or wrong. "The answer was bad" is unactionable;
"retrieval took 12ms and returned nothing relevant, so generation had nothing
to work with" tells you precisely where to look.

So every query is timed per stage, and those timings travel out with the
response rather than dying in a log line. That single habit is what turns a
pipeline into something operable: latency budgets become measurable per block,
and a regression in reranking is visibly a reranking regression.

This module deliberately depends on nothing but the standard library. In a
real deployment these spans map one-to-one onto OpenTelemetry spans (or
Langfuse/Phoenix traces) -- `Stopwatch.stage()` is the seam where you'd swap
`time.perf_counter()` for `tracer.start_as_current_span()` without touching a
single call site.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Dict, Generator


def configure_logging(level: str = "INFO") -> None:
    """Idempotent logging setup, safe to call from both the API and the CLI."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries log a progress bar's worth of noise at INFO.
    for noisy in ("httpx", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Stopwatch:
    """Accumulates per-stage wall-clock timings for a single request."""

    def __init__(self) -> None:
        self.timings_ms: Dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - start) * 1000, 2)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 2)

    def as_dict(self) -> Dict[str, float]:
        return {**self.timings_ms, "total": self.total_ms()}
