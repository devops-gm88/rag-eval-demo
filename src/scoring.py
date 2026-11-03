"""Per-question scoring for the eval harness.

Kept separate from the Postgres evaluation runner so the scaled-down demo can
score with the same rules without importing the database layer.
"""
from __future__ import annotations

from .matching import first_match


def score_answerable(item: dict, result: dict) -> dict:
    sources = [chunk["source"] for chunk in result["chunks"]]
    recall = any(source in item["expected_sources"] for source in sources)

    fact_forms = [item["key_fact"], *item.get("aliases", [])]
    cited_chunks = [
        result["chunks"][index - 1]
        for index in result["cited"]
        if 1 <= index <= len(result["chunks"])
    ]
    citation_fact = None
    for chunk in cited_chunks:
        citation_fact = first_match(chunk["text"], fact_forms)
        if citation_fact:
            break
    citation = citation_fact is not None and not result["invalid_citations"]
    answer_fact = first_match(result["answer"], fact_forms)
    complete = answer_fact is not None

    return {
        "recall_at_5": recall,
        "citation_correct": citation,
        "answer_complete": complete,
        "passed": bool(recall and citation and complete),
        "citation_fact": citation_fact,
        "answer_fact": answer_fact,
        "retrieved_sources": sources,
    }
