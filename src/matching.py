"""Normalisation and matching for the eval metrics."""
from __future__ import annotations

import re

PUNCT = re.compile(r"[^a-z0-9./%+#\- ]+")
ARTICLES = {"the", "a", "an"}


def normalise(text: str) -> str:
    """Lowercase, drop currency and thousands separators, drop articles.

    Dropping articles makes "2 of 5" match "2 of the 5 cases". The articles
    carry no meaning in a fact check.
    """
    text = text.lower()
    text = text.replace("$", "").replace(",", "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = PUNCT.sub(" ", text)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def contains(haystack: str, needle: str) -> bool:
    needle_norm = normalise(needle)
    if not needle_norm:
        return False
    return needle_norm in normalise(haystack)


def first_match(haystack: str, facts: list[str]) -> str | None:
    for fact in facts:
        if contains(haystack, fact):
            return fact
    return None
