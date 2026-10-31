"""Chunk, embed and load the corpus into both indexes.

    python -m src.ingest              # both chunkings
    python -m src.ingest --chunking structural
"""
from __future__ import annotations

import argparse
import json
import time

from .chunking import chunk_documents
from .config import CORPUS_DIR
from .corpus import load_documents
from .db import Database
from .embedding import Ollama


def ingest(chunking: str, db: Database, ollama: Ollama, docs: list[dict] | None = None) -> dict:
    docs = docs if docs is not None else load_documents()
    started = time.perf_counter()
    chunks = chunk_documents(docs, chunking)
    embed_started = time.perf_counter()
    vectors = ollama.embed_documents([chunk["text"] for chunk in chunks])
    embed_ms = round((time.perf_counter() - embed_started) * 1000, 1)

    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector

    db.reset(chunking)
    inserted = db.insert_chunks(chunking, chunks)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {
        "chunking": chunking,
        "documents": len(docs),
        "chunks": inserted,
        "embed_ms": embed_ms,
        "total_ms": elapsed,
        "chars": sum(len(chunk["text"]) for chunk in chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking", choices=["fixed", "structural", "both"], default="both")
    args = parser.parse_args()

    db = Database().connect()
    db.init_schema()
    ollama = Ollama()
    health = ollama.health()
    if not health.get("embed_model_present"):
        raise SystemExit(f"embedding model missing: {health}. Run: ollama pull {ollama.embed_model}")

    modes = ["fixed", "structural"] if args.chunking == "both" else [args.chunking]
    results = []
    for mode in modes:
        result = ingest(mode, db, ollama)
        results.append(result)
        print(json.dumps(result))

    summary = {
        "corpus": str(CORPUS_DIR / "documents.jsonl"),
        "documents": len(load_documents()),
        "indexes": {result["chunking"]: result["chunks"] for result in results},
        "counts": {mode: db.count(mode) for mode in ("fixed", "structural")},
    }
    (CORPUS_DIR / "ingest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
