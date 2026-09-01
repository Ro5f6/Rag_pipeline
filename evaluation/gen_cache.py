"""
evaluation/gen_cache.py
=======================
A tiny on-disk cache of generated answers, keyed by everything that affects the
output: the generator (backend + model), the question, and the retrieval
settings. Generation is the slow, rate-limited, costly part of a RAGAS run;
caching it means tweaking metrics or re-running the judge never pays to
regenerate the same answers.

Mirrors the pipeline's persisted-index philosophy: compute once, reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

CACHE_PATH = Path("data/eval/.gen_cache.json")


def _key(backend: str, model: str, question_id: str, mode: str, rerank: bool, top_k: int) -> str:
    return f"{backend}|{model}|{question_id}|{mode}|rerank={rerank}|k={top_k}"


class AnswerCache:
    """Maps a generation key -> {"answer": str, "contexts": [str, ...]}."""

    def __init__(self, path: Path = CACHE_PATH, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._data: Dict[str, dict] = {}
        if enabled and path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, backend, model, question_id, mode, rerank, top_k) -> Optional[dict]:
        if not self.enabled:
            return None
        return self._data.get(_key(backend, model, question_id, mode, rerank, top_k))

    def put(self, backend, model, question_id, mode, rerank, top_k, answer: str, contexts: List[str]) -> None:
        if not self.enabled:
            return
        self._data[_key(backend, model, question_id, mode, rerank, top_k)] = {
            "answer": answer,
            "contexts": contexts,
        }

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
