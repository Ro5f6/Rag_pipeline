"""
pipeline/context_format.py
=============================
FLOWCHART BLOCK: "Context format"

Responsibility: turn the final reranked chunks into a single well-structured
string to inject into the LLM prompt -- and, critically, tag each chunk with
a source id so the LLM can cite exactly which chunk it used later.

This block matters more than it looks. LLMs are sensitive to prompt
structure: unlabeled walls of text lead to hallucinated or unattributable
answers. Numbering sources explicitly ([1], [2], ...) and instructing the
model to cite them is what makes the "Cite sources" block downstream
actually work.
"""

from typing import List
from pipeline.rerank import RerankedChunk


def format_context(reranked_chunks: List[RerankedChunk]) -> str:
    """
    Produces something like:

    [1] (source: rag_basics.txt)
    Retrieval Augmented Generation, or RAG, combines a retrieval system...

    [2] (source: vllm_overview.txt)
    vLLM is a high-throughput and memory-efficient inference engine...
    """
    blocks = []
    for i, rc in enumerate(reranked_chunks, start=1):
        filename = rc.chunk.metadata.get("filename", rc.chunk.source)
        blocks.append(f"[{i}] (source: {filename})\n{rc.chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    """
    The instruction telling the LLM HOW to use citations lives here, right
    next to where the context is built -- keeping "what goes into the
    prompt" in one place makes prompt-engineering iteration much easier.
    """
    return f"""You are a helpful assistant answering questions using ONLY the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer using only information from the context above.
- Cite sources inline using the bracket numbers, e.g. [1], [2].
- If the context does not contain the answer, say so explicitly -- do not guess.

Answer:"""
