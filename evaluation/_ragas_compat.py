"""
evaluation/_ragas_compat.py
============================
Makes RAGAS 0.4.x importable on the LangChain 1.x stack this project uses.

RAGAS 0.4.3 still hard-imports `ChatVertexAI` from `langchain_community.chat_models.vertexai`
at module load. `langchain-community` 0.4.x (now sunset) removed that path -- the
Vertex integration moved to the standalone `langchain-google-vertexai` package.
The import therefore fails before RAGAS is even usable.

This project uses Gemini and Groq, never Vertex, so rather than pin an older
langchain stack we install a lightweight placeholder for the dead import. The
placeholder only raises if someone actually tries to construct a Vertex model,
which this project never does.

Import this module *before* importing `ragas` (eval/ragas_judge.py does).
"""

from __future__ import annotations

import sys
import types
import warnings

# RAGAS 0.4.x imports langchain_community at load, which emits an unactionable
# "being sunset" DeprecationWarning. We can't avoid the import, so silence just
# that message rather than let it clutter every eval run.
warnings.filterwarnings("ignore", message=r".*langchain-community.*")


class _VertexNotConfigured:
    """Stand-in for ChatVertexAI / VertexAI -- never constructed in this project."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Vertex AI is not configured in this project. "
            "Use RAG_JUDGE_PROVIDER=google_genai (Gemini) or another supported provider."
        )


def install() -> None:
    """Idempotently satisfy the Vertex imports RAGAS 0.4.x requires."""
    modname = "langchain_community.chat_models.vertexai"
    if modname not in sys.modules:
        module = types.ModuleType(modname)
        module.ChatVertexAI = _VertexNotConfigured
        sys.modules[modname] = module

    # `from langchain_community.llms import VertexAI` -- present in current
    # community builds, but shim it defensively if a future version drops it too.
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:  # pragma: no cover - only hit if community also removes it
        import langchain_community.llms as _llms

        _llms.VertexAI = _VertexNotConfigured


install()
