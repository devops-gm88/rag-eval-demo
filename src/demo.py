"""Scaled-down, Postgres-free demo of the eval harness.

Runs the shipped structural chunker, the same answer prompt and the same four
metrics over 13 synthetic documents and 10 questions (8 answerable, 2 refusals).
The index is an in-memory cosine table, so the demo needs no database.

    python -m src.demo              # run and score
    python -m src.demo --check-only # re-score the committed results, no model calls

Writes reports/demo_results.json and exits non-zero when a metric in
evals/demo_gates.json is not met.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import date

from .answering import REFUSAL, answer_question, is_refusal
from .chunking import structural_chunks
from .config import CHAT_MODEL, EMBED_MODEL, EVALS_DIR, REPORTS_DIR, ROOT, TOP_K
from .embedding import Ollama
from .scoring import score_answerable

DEMO_DIR = ROOT / "data" / "demo_corpus"
QUESTIONS_PATH = EVALS_DIR / "demo_questions.json"
GATES_PATH = EVALS_DIR / "demo_gates.json"
RESULTS_PATH = REPORTS_DIR / "demo_results.json"

GATE_METRICS = ("recall_at_5", "citation_correct", "answer_complete", "passed", "refusals_correct")


def load_documents() -> list[dict]:
    documents = []
    for path in sorted(DEMO_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        first = next((line for line in text.splitlines() if line.startswith("# ")), path.stem)
        documents.append(
            {
                "source": f"demo_corpus/{path.name}",
                "title": first[2:].strip(),
                "kind": "md",
                "text": text,
            }
        )
    return documents


def load_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cosine(a: list[float], b: list[float]) -> float:
    return dot(a, b) / (math.sqrt(dot(a, a)) * math.sqrt(dot(b, b)) or 1.0)


def retrieve(chunks: list[dict], vectors: list[list[float]], query_vector: list[float], k: int = TOP_K) -> list[dict]:
    scored = sorted(
        ({"score": cosine(query_vector, vector), **chunk} for chunk, vector in zip(chunks, vectors, strict=True)),
        key=lambda row: row["score"],
        reverse=True,
    )[:k]
    for position, row in enumerate(scored, start=1):
        row["rank"] = position
    return scored


def validate(questions: dict, documents: list[dict]) -> None:
    sources = {document["source"] for document in documents}
    assert len(documents) >= 12, f"expected at least 12 demo documents, found {len(documents)}"
    for item in questions["answerable"]:
        assert item["expected_sources"], f"{item['id']} has no expected source"
        missing = [source for source in item["expected_sources"] if source not in sources]
        assert not missing, f"{item['id']} names sources that do not exist: {missing}"
    assert len(questions["answerable"]) == 8, "demo set must have 8 answerable questions"
    assert len(questions["unanswerable"]) == 2, "demo set must have 2 refusal questions"


def gate_failures(totals: dict, gates: dict) -> list[str]:
    return [
        f"{metric}: {totals.get(metric, 0)} < gate {gates[metric]}"
        for metric in GATE_METRICS
        if metric in gates and totals.get(metric, 0) < gates[metric]
    ]


def print_summary(totals: dict, gates: dict, failures: list[str], wall_seconds: float) -> None:
    print(
        f"queries: recall@5 {totals['recall_at_5']}/{totals['questions']}  "
        f"citation {totals['citation_correct']}/{totals['questions']}  "
        f"complete {totals['answer_complete']}/{totals['questions']}  "
        f"passed {totals['passed']}/{totals['questions']}  "
        f"refusals {totals['refusals_correct']}/{totals['unanswerable']}"
    )
    if failures:
        for failure in failures:
            print(f"REGRESSION {failure}")
        print("demo FAILED: a metric is below its gate in evals/demo_gates.json")
    else:
        print(f"demo passed every gate in evals/demo_gates.json ({wall_seconds:.1f}s)")


def run() -> int:
    started = time.perf_counter()
    ollama = Ollama()
    health = ollama.health()
    if not health.get("reachable"):
        print(f"Ollama is not reachable at {ollama.url}: {health.get('error')}", file=sys.stderr)
        return 2
    if not health.get("embed_model_present") or not health.get("chat_model_present"):
        print(
            f"missing model: embed={EMBED_MODEL} present={health.get('embed_model_present')}, "
            f"chat={CHAT_MODEL} present={health.get('chat_model_present')}",
            file=sys.stderr,
        )
        return 2

    documents = load_documents()
    questions = load_json(QUESTIONS_PATH)
    gates = load_json(GATES_PATH)
    validate(questions, documents)

    chunks: list[dict] = []
    for document in documents:
        chunks.extend(structural_chunks(document))
    print(f"{len(documents)} documents -> {len(chunks)} chunks", flush=True)

    # Embed documents and questions before any generation, so Ollama does not
    # swap the chat and embedding models between calls.
    document_vectors = ollama.embed_documents([chunk["text"] for chunk in chunks])
    items = [*questions["answerable"], *questions["unanswerable"]]
    query_vectors = dict(
        zip(
            [item["id"] for item in items],
            ollama.embed([f"search_query: {item['question']}" for item in items]),
            strict=True,
        )
    )
    print(f"embedded {len(items)} questions", flush=True)

    rows = []
    for item in questions["answerable"]:
        retrieval = retrieve(chunks, document_vectors, query_vectors[item["id"]])
        answer = answer_question(ollama, item["question"], retrieval)
        scores = score_answerable(item, {**{"chunks": retrieval}, **answer})
        status = "PASS" if scores["passed"] else "FAIL"
        print(f"  {item['id']} {status} {item['question'][:70]}", flush=True)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_sources": item["expected_sources"],
                "answer": answer["answer"],
                "cited": answer["cited"],
                "invalid_citations": answer["invalid_citations"],
                "refused": answer["refused"],
                "retrieved": [
                    {"rank": row["rank"], "source": row["source"], "chunk_index": row["chunk_index"], "score": round(row["score"], 5)}
                    for row in retrieval
                ],
                **scores,
            }
        )

    refusals = []
    for item in questions["unanswerable"]:
        retrieval = retrieve(chunks, document_vectors, query_vectors[item["id"]])
        answer = answer_question(ollama, item["question"], retrieval)
        correct = bool(is_refusal(answer["answer"]))
        status = "PASS" if correct else "FAIL"
        print(f"  {item['id']} {status} {item['question'][:70]}", flush=True)
        refusals.append(
            {
                "id": item["id"],
                "question": item["question"],
                "why": item.get("why", ""),
                "answer": answer["answer"],
                "refused": answer["refused"],
                "correct": correct,
                "retrieved": [row["source"] for row in retrieval],
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
    failures = gate_failures(totals, gates)
    wall_seconds = round(time.perf_counter() - started, 1)

    result = {
        "suite": "demo",
        "date": str(date.today()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "documents": len(documents),
            "chunks": len(chunks),
            "top_k": TOP_K,
            "refusal_token": REFUSAL,
        },
        "gates": gates,
        "gate_failures": failures,
        "totals": totals,
        "wall_seconds": wall_seconds,
        "rows": rows,
        "refusals": refusals,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(totals, gates, failures, wall_seconds)
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)}")
    return 1 if failures else 0


def check_only() -> int:
    if not RESULTS_PATH.exists():
        print(f"{RESULTS_PATH} does not exist; run `make demo` first", file=sys.stderr)
        return 2
    result = load_json(RESULTS_PATH)
    gates = load_json(GATES_PATH)
    failures = gate_failures(result["totals"], gates)
    print_summary(result["totals"], gates, failures, result.get("wall_seconds", 0.0))
    print(f"checked {RESULTS_PATH.relative_to(ROOT)}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="re-score the committed results without calling the models")
    args = parser.parse_args()
    raise SystemExit(check_only() if args.check_only else run())


if __name__ == "__main__":
    main()
