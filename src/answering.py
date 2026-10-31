"""Answer generation, citing rules and the refusal path."""
from __future__ import annotations

import re

from .embedding import Ollama

REFUSAL = "NOT IN SOURCES"

SYSTEM = (
    "You answer questions about a private document set. "
    "You never invent facts and you never use outside knowledge."
)

PROMPT = """Answer the question from the sources below.

Rules:
1. Put the source number in square brackets after every claim, for example [2].
2. If the sources do not contain the answer, reply exactly: NOT IN SOURCES
3. Use only the sources. Do not add outside knowledge.
4. Answer in one to three sentences.

{sources}

QUESTION: {question}

ANSWER:"""

MAX_CHUNK_CHARS = 1200


def build_prompt(question: str, chunks: list[dict]) -> str:
    parts = []
    for position, chunk in enumerate(chunks, start=1):
        title = (chunk.get("meta") or {}).get("title", "")
        body = chunk["text"][:MAX_CHUNK_CHARS]
        parts.append(f"SOURCE {position} (title: {title})\n{body}")
    return PROMPT.format(sources="\n\n".join(parts), question=question)


def parse_citations(answer: str, chunk_count: int) -> dict:
    """Accept [2] and grouped forms such as [1, 2, 4]."""
    ids: list[int] = []
    for group in re.findall(r"\[([0-9][0-9,\s]*)\]", answer):
        ids.extend(int(value) for value in re.findall(r"\d{1,2}", group))
    valid = sorted({value for value in ids if 1 <= value <= chunk_count})
    invalid = sorted({value for value in ids if value < 1 or value > chunk_count})
    return {"cited": valid, "invalid": invalid}


def is_refusal(answer: str) -> bool:
    return "NOT IN SOURCES" in answer.upper()


def answer_question(ollama: Ollama, question: str, chunks: list[dict]) -> dict:
    prompt = build_prompt(question, chunks)
    answer = ollama.chat(prompt, system=SYSTEM)
    citations = parse_citations(answer, len(chunks))
    return {
        "answer": answer,
        "refused": is_refusal(answer),
        "cited": citations["cited"],
        "invalid_citations": citations["invalid"],
        "prompt_chars": len(prompt),
    }
