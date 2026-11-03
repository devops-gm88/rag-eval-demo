"""Run the eval over four configurations and write JSON reports.

    python -m src.evaluate                # all four configurations
    python -m src.evaluate --configs baseline

Configurations (chunking x retrieval):
    baseline            fixed 500-token chunks  + vector only
    fixed_hybrid        fixed chunks            + full-text hybrid
    structural_vector   structural chunks       + vector only
    improved            structural chunks       + full-text hybrid

Metrics per answerable question:
    recall@5        an expected source is in the top five chunks
    citation        a cited chunk contains the key fact
    complete        the answer contains the key fact
    passed          recall and citation and complete
Unanswerable questions (3) must return NOT IN SOURCES.
Pass bar: 26 of 30 answered correctly and 3 of 3 refusals.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date
from pathlib import Path

from .answering import REFUSAL, answer_question
from .config import CHAT_MODEL, EMBED_MODEL, EVALS_DIR, REPORTS_DIR, TOP_K
from .corpus import load_documents
from .db import Database
from .embedding import Ollama
from .retrieval import retrieve
from .scoring import score_answerable

CONFIGS = {
    "baseline": {"chunking": "fixed", "mode": "vector"},
    "fixed_hybrid": {"chunking": "fixed", "mode": "hybrid"},
    "structural_vector": {"chunking": "structural", "mode": "vector"},
    "improved": {"chunking": "structural", "mode": "hybrid"},
}

OUTPUT_NAMES = {
    "baseline": "eval_baseline.json",
    "fixed_hybrid": "eval_fixed_hybrid.json",
    "structural_vector": "eval_structural_vector.json",
    "improved": "eval_improved.json",
}

PASS_BAR = {"answered": 26, "refusals": 3}


def load_questions() -> dict:
    return json.loads((EVALS_DIR / "questions.json").read_text(encoding="utf-8"))


def run_config(
    name: str,
    config: dict,
    questions: dict,
    db: Database,
    ollama: Ollama,
    vectors: dict[str, list[float]],
) -> dict:
    rows = []
    started = time.perf_counter()
    for item in questions["answerable"]:
        retrieval = retrieve(
            db, ollama, item["question"], config["chunking"], config["mode"], TOP_K,
            query_vector=vectors[item["id"]],
        )
        answer = answer_question(ollama, item["question"], retrieval["chunks"])
        scores = score_answerable(item, {**retrieval, **answer})
        print(f"  {item['id']} {scores['passed'] and 'PASS' or 'FAIL'} {retrieval['retrieval_ms']:.0f}ms", flush=True)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_sources": item["expected_sources"],
                "key_fact": item["key_fact"],
                "answer": answer["answer"],
                "cited": answer["cited"],
                "invalid_citations": answer["invalid_citations"],
                "refused": answer["refused"],
                "retrieval_ms": retrieval["retrieval_ms"],
                "retrieved": [
                    {
                        "rank": chunk["rank"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "score": round(float(chunk.get("score") or 0), 5),
                    }
                    for chunk in retrieval["chunks"]
                ],
                **scores,
            }
        )

    refusals = []
    for item in questions["unanswerable"]:
        retrieval = retrieve(
            db, ollama, item["question"], config["chunking"], config["mode"], TOP_K,
            query_vector=vectors[item["id"]],
        )
        answer = answer_question(ollama, item["question"], retrieval["chunks"])
        refusals.append(
            {
                "id": item["id"],
                "question": item["question"],
                "why": item.get("why", ""),
                "answer": answer["answer"],
                "refused": answer["refused"],
                "correct": bool(answer["refused"]),
                "retrieved": [chunk["source"] for chunk in retrieval["chunks"]],
            }
        )

    totals = {
        "recall_at_5": sum(row["recall_at_5"] for row in rows),
        "citation_correct": sum(row["citation_correct"] for row in rows),
        "answer_complete": sum(row["answer_complete"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "refusals_correct": sum(row["correct"] for row in refusals),
        "questions": len(rows),
        "unanswerable": len(refusals),
    }
    totals["pass_bar_met"] = totals["passed"] >= PASS_BAR["answered"] and totals["refusals_correct"] >= PASS_BAR["refusals"]
    totals["mean_retrieval_ms"] = round(sum(row["retrieval_ms"] for row in rows) / len(rows), 1)

    return {
        "config": name,
        "chunking": config["chunking"],
        "retrieval_mode": config["mode"],
        "models": {"embedding": EMBED_MODEL, "chat": CHAT_MODEL},
        "top_k": TOP_K,
        "pass_bar": PASS_BAR,
        "totals": totals,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
        "refusals": refusals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = parser.parse_args()

    questions = load_questions()
    db = Database().connect()
    db.init_schema()
    ollama = Ollama()

    # Embed every question once, before any generation. Interleaving embedding and
    # chat calls makes Ollama reload a 4.7 GB model on every question.
    all_questions = [*questions["answerable"], *questions["unanswerable"]]
    embedded = ollama.embed([f"search_query: {item['question']}" for item in all_questions])
    vectors = {item["id"]: vector for item, vector in zip(all_questions, embedded, strict=True)}
    print(f"embedded {len(vectors)} questions")
    REPORTS_DIR.mkdir(exist_ok=True)

    environment = {
        "date": str(date.today()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedding_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "documents": len(load_documents()),
        "chunks_structural": db.count("structural"),
        "chunks_fixed": db.count("fixed"),
        "refusal_token": REFUSAL,
    }

    for name in args.configs:
        result = run_config(name, CONFIGS[name], questions, db, ollama, vectors)
        result["environment"] = environment
        path = REPORTS_DIR / OUTPUT_NAMES[name]
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        totals = result["totals"]
        print(
            f"{name:18s} recall@5 {totals['recall_at_5']}/{totals['questions']}  "
            f"citation {totals['citation_correct']}/{totals['questions']}  "
            f"complete {totals['answer_complete']}/{totals['questions']}  "
            f"passed {totals['passed']}/{totals['questions']}  "
            f"refusals {totals['refusals_correct']}/{totals['unanswerable']}  "
            f"({result['wall_seconds']}s) -> {path.name}"
        )


if __name__ == "__main__":
    main()
