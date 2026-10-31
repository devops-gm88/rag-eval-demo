"""Two retrieval modes over the two chunkings.

vector   cosine top-k on the pgvector index
hybrid   vector top-N plus Postgres full-text top-N, fused with reciprocal rank
         fusion (RRF, k=60). This is the "Postgres full-text hybrid retrieval"
         option from the brief; there is no rerank model in the loop.
"""
from __future__ import annotations

import time

from .config import HYBRID_CANDIDATES, RRF_K, TOP_K
from .db import Database
from .embedding import Ollama


def reciprocal_rank_fusion(
    lists: list[list[dict]], k: int = RRF_K, limit: int = TOP_K
) -> list[dict]:
    scores: dict[int, float] = {}
    best: dict[int, dict] = {}
    for ranked in lists:
        for rank, row in enumerate(ranked, start=1):
            key = int(row["id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            stored = best.get(key)
            if stored is None:
                best[key] = dict(row)
            # keep the higher of the two component scores for reporting
            if row.get("score") is not None and (stored or {}).get("score") is not None:
                best[key]["score"] = max(stored["score"], row["score"])
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    fused: list[dict] = []
    for position, (key, fused_score) in enumerate(ordered, start=1):
        row = dict(best[key])
        row["rrf_score"] = round(fused_score, 6)
        row["rank"] = position
        fused.append(row)
    return fused


def retrieve(
    db: Database,
    ollama: Ollama,
    question: str,
    chunking: str,
    mode: str,
    k: int = TOP_K,
    query_vector: list[float] | None = None,
) -> dict:
    started = time.perf_counter()
    if query_vector is None:
        query_vector = ollama.embed_query(question)

    if mode == "vector":
        rows = db.vector_search(chunking, query_vector, k)
        for position, row in enumerate(rows, start=1):
            row["rank"] = position
        candidates = len(rows)
    elif mode == "hybrid":
        vector_rows = db.vector_search(chunking, query_vector, HYBRID_CANDIDATES)
        fts_rows = db.fts_search(chunking, question, HYBRID_CANDIDATES)
        rows = reciprocal_rank_fusion([vector_rows, fts_rows], limit=k)
        candidates = len(vector_rows) + len(fts_rows)
    else:
        raise ValueError(f"unknown retrieval mode {mode!r}")

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "chunks": rows,
        "retrieval_ms": elapsed_ms,
        "candidates": candidates,
        "mode": mode,
        "chunking": chunking,
    }
