"""Unit tests for the chunkers, matching and fusion. No model, no database."""
from __future__ import annotations

import json

import pytest

from src.chunking import baseline_chunks, chunk_documents, estimate_tokens, structural_chunks
from src.corpus import html_to_text, redact
from src.evaluate import score_answerable
from src.matching import contains, first_match, normalise
from src.retrieval import reciprocal_rank_fusion

DOC = {
    "source": "demo.md#Scope",
    "title": "Scope",
    "kind": "md",
    "text": "# Scope\n\nThe pilot covers 120 vehicles.\n\n## Timing\n\nLive position every 15 minutes.",
    "chars": 90,
}


def test_baseline_chunks_respect_the_token_budget():
    long_text = "word " * 900
    doc = {**DOC, "text": long_text}
    chunks = baseline_chunks(doc)
    assert len(chunks) > 1
    assert all(estimate_tokens(chunk["text"]) <= 520 for chunk in chunks)


def test_baseline_chunks_are_sequential_and_keyed():
    doc = {**DOC, "text": "word " * 900}
    chunks = baseline_chunks(doc)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["source"] == doc["source"] for chunk in chunks)


def test_structural_chunks_carry_the_heading_path():
    chunks = structural_chunks(DOC)
    assert chunks
    assert all(chunk["text"].startswith("Scope") for chunk in chunks)
    assert any("Timing" in chunk["text"] for chunk in chunks)


def test_structural_chunks_stay_under_the_budget():
    doc = {**DOC, "text": "# Big\n\n" + ("sentence. " * 800)}
    chunks = structural_chunks(doc)
    assert all(estimate_tokens(chunk["text"]) <= 520 for chunk in chunks)


def test_chunk_documents_switches_mode():
    fixed = chunk_documents([DOC], "fixed")
    structural = chunk_documents([DOC], "structural")
    assert fixed[0]["meta"]["chunker"] == "fixed"
    assert structural[0]["meta"]["chunker"] == "structural"


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 100) < estimate_tokens("a" * 200)


def test_normalise_strips_currency_and_commas():
    assert normalise("$1,245 jobs") == "1245 jobs"


def test_contains_matches_across_surface_forms():
    assert contains("The materiality threshold is 50 CAD.", "50 CAD")
    assert contains("We compared 41 ledger rows.", "41 ledger rows")
    assert not contains("The answer is 120.", "120 vehicles")


def test_first_match_returns_the_first_alias_found():
    assert first_match("The total is 2 of 5 cases.", ["2 of 5", "two of five"]) == "2 of 5"
    assert first_match("nothing here", ["2 of 5"]) is None


def test_rrf_puts_a_row_in_both_lists_first():
    left = [{"id": 1}, {"id": 2}]
    right = [{"id": 2}, {"id": 3}]
    fused = reciprocal_rank_fusion([left, right], limit=3)
    assert fused[0]["id"] == 2
    assert [row["rank"] for row in fused] == [1, 2, 3]


def test_rrf_respects_the_limit():
    rows = [{"id": index} for index in range(10)]
    assert len(reciprocal_rank_fusion([rows], limit=4)) == 4


def test_html_to_text_keeps_headings():
    text = html_to_text("<h1>Title</h1><p>Body copy here.</p><ul><li>One</li></ul>")
    assert "# Title" in text
    assert "Body copy here." in text
    assert "- One" in text


def test_redact_removes_names_and_emails():
    out = redact("Natalie wrote to owen@example.com about Owen's plan.")
    assert "Natalie" not in out
    assert "owen@example.com" not in out
    assert "Owen" not in out


def test_score_answerable_requires_all_three_checks():
    item = {"key_fact": "120 vehicles", "aliases": [], "expected_sources": ["demo.md#Scope"]}
    chunk = {"source": "demo.md#Scope", "text": "The pilot covers 120 vehicles.", "chunk_index": 0}
    good = {
        "chunks": [chunk],
        "cited": [1],
        "invalid_citations": [],
        "answer": "The pilot covers 120 vehicles [1].",
    }
    assert score_answerable(item, good)["passed"] is True

    wrong_source = {**good, "chunks": [{**chunk, "source": "other.md"}]}
    assert score_answerable(item, wrong_source)["passed"] is False

    no_citation = {**good, "cited": []}
    assert score_answerable(item, no_citation)["citation_correct"] is False


def test_questions_file_is_well_formed():
    questions = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "evals" / "questions.json").read_text())
    assert len(questions["answerable"]) == 30
    assert len(questions["unanswerable"]) == 3
    ids = [item["id"] for item in questions["answerable"] + questions["unanswerable"]]
    assert len(ids) == len(set(ids))
    for item in questions["answerable"]:
        assert item["expected_sources"], item["id"]
        assert item["key_fact"], item["id"]
