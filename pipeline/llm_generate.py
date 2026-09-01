"""
pipeline/llm_generate.py
==========================
FLOWCHART BLOCK: "LLM generate"

Responsibility: turn the formatted prompt (query + retrieved context) into a
grounded answer. This is the only block in the pipeline that talks to a model
provider -- everything before it is retrieval engineering, everything after it
is post-processing.

DESIGN: one interface, any model, via LangChain's init_chat_model.
--------------------------------------------------------------------------
Nothing upstream or downstream of this file knows which model answers the
question. There is a single generator class -- ChatModelGenerator -- that wraps
any LangChain chat model behind one method:

    generate(prompt, *, query, chunks) -> str

`init_chat_model` builds the underlying client from a provider id and a model
id, so adding a provider is a config change, not a code change. The same code
path serves hosted APIs and self-hosted open-source servers:

    groq          Llama / Qwen / Mixtral on Groq's fast hosted endpoint
    ollama        any local model served by Ollama
    openai        OpenAI *and* every OpenAI-compatible server (vLLM, TGI,
                  LMStudio, Together, Fireworks, OpenRouter) via base_url
    google_genai  Gemini
    anthropic     Claude

Provider is read from `RAG_LLM_PROVIDER` as a canonical LangChain id
(`groq`, `ollama`, `openai`, `google_genai`, `anthropic`), or left as `auto`
to infer it from the model name.

This project runs generation through a real model on purpose -- there is no
offline fallback. Tests stay hostless by injecting a fake chat model (see
tests/), which is still the same BaseChatModel interface, just deterministic.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# Reasoning models (e.g. Qwen "thinking" variants) emit their chain-of-thought
# inline, wrapped in <think>...</think>, before the actual answer. That text must
# not reach the user, the citation mapper, or the evaluation judge.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning, leaving only the user-facing answer."""
    text = _THINK_BLOCK.sub("", text)
    # Handle an unpaired closing tag (reasoning left open, or only </think> echoed):
    # keep whatever follows the last one.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


class Generator(Protocol):
    """The one interface the pipeline depends on."""

    name: str
    model: str

    def generate(self, prompt: str, *, query: str, chunks: Sequence) -> str:
        ...


# The one knob whose name is genuinely provider-specific. Keyed on LangChain's
# canonical provider ids.
_MAX_TOKENS_KWARG = {
    "google_genai": "max_output_tokens",
    "ollama": "num_predict",
}


def _resolve_provider(provider: str) -> Optional[str]:
    """Normalize the configured provider, or return None to let
    init_chat_model infer it from a well-known model prefix (gpt-*, claude-*).

    Use LangChain's canonical ids directly (google_genai, openai, groq, ollama,
    anthropic); OpenAI-compatible gateways use `openai` with a base_url."""
    provider = (provider or "").strip().lower()
    if not provider or provider == "auto":
        return None
    return provider


class ChatModelGenerator:
    """
    Wraps any LangChain chat model behind the pipeline's Generator interface.

    The chunks are already baked into `prompt` by the context formatter; they
    are accepted here only to satisfy the shared interface (the citation mapper
    downstream is what actually uses them).
    """

    def __init__(self, chat_model, *, provider: str, model: str):
        self._chat = chat_model
        self.name = provider
        self.model = model

    def generate(self, prompt: str, *, query: str = "", chunks: Sequence = ()) -> str:
        message = self._chat.invoke(prompt)
        content = getattr(message, "content", message)

        # A chat model returns either a plain string or a list of content
        # blocks (dicts with a "text" field, or objects). Flatten to text.
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    parts.append(getattr(part, "text", str(part)))
            content = "".join(parts)

        return _strip_reasoning(content or "")


def build_generator(
    provider: str = "auto",
    model: str = "",
    max_tokens: int = 1024,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Generator:
    """
    Build the configured generation backend via LangChain's init_chat_model.

    `model` is required -- there is no offline default. `provider` is a canonical
    LangChain id (groq, ollama, openai, google_genai, anthropic), or 'auto' to
    infer the provider from the model name (works for gpt-*, claude-*, gemini-*).
    OpenAI-compatible servers use provider='openai' with a base_url.
    """
    from langchain.chat_models import init_chat_model

    if not model:
        raise ValueError(
            "No model configured. Set RAG_LLM_MODEL (e.g. 'llama-3.3-70b-versatile' "
            "with RAG_LLM_PROVIDER=groq, or 'gemini-2.5-flash' with provider=google_genai)."
        )

    resolved = _resolve_provider(provider)

    kwargs = {
        "model_provider": resolved,
        "temperature": temperature,
        "max_retries": max_retries,
    }
    kwargs[_MAX_TOKENS_KWARG.get(resolved, "max_tokens")] = max_tokens
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        chat = init_chat_model(model, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised as a clear config error
        raise ValueError(
            f"Could not initialise a chat model (provider={resolved or 'inferred'!r}, "
            f"model={model!r}): {exc}. Check RAG_LLM_PROVIDER / RAG_LLM_MODEL and that "
            f"the matching langchain-<provider> package is installed."
        ) from exc

    logger.info("Generation via LangChain: provider=%s model=%s", resolved or "inferred", model)
    return ChatModelGenerator(chat, provider=resolved or "auto", model=model)
