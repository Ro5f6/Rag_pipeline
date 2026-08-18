"""
pipeline/query_rewrite.py
==========================
FLOWCHART BLOCK: "Query rewrite"

Responsibility: transform the raw user query into a form that retrieves
better. Real user queries are often short, ambiguous, or conversational
("what about the memory thing again?") -- retrieval quality improves a lot
if you clean/expand the query before embedding or BM25-matching it.

Three common strategies, from cheapest to most powerful:
1. Rule-based cleanup (strip filler words, expand known acronyms).
2. LLM-based rewrite: ask a cheap/fast LLM call to restate the query more
   precisely, or to generate the *hypothetical answer* and embed that
   instead (this technique is called HyDE -- Hypothetical Document
   Embeddings -- and often improves recall).
3. Multi-query expansion: ask the LLM for 3 paraphrases of the query, run
   retrieval for all of them, and merge results -- useful when one phrasing
   might miss relevant chunks that a different phrasing would catch.

We implement (1) here for zero-dependency clarity, and stub (2) so you can
wire in a real LLM call once you have your API key set up.
"""

import re

_ACRONYM_EXPANSIONS = {
    "kv cache": "key-value cache",
    "rag": "retrieval augmented generation",
    "llm": "large language model",
}


def rule_based_rewrite(query: str) -> str:
    """Cheap, deterministic, no LLM call needed."""
    cleaned = query.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for acronym, expansion in _ACRONYM_EXPANSIONS.items():
        if acronym in cleaned and expansion not in cleaned:
            cleaned = f"{cleaned} ({expansion})"

    return cleaned


def llm_based_rewrite(query: str, llm_call_fn) -> str:
    """
    Optional upgrade: pass in any callable(str) -> str that hits an LLM.
    Kept as a thin wrapper so orchestrator.py can choose rule_based_rewrite
    or llm_based_rewrite via config without changing its own code.
    """
    prompt = (
        "Rewrite the following user question to be maximally clear and "
        "specific for a document search engine. Return ONLY the rewritten "
        f"question, nothing else.\n\nQuestion: {query}"
    )
    return llm_call_fn(prompt).strip()
