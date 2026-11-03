# rag-eval-demo

A retrieval-augmented answering pipeline with a **published evaluation set**: 30 questions,
3 questions that must be refused, four configurations, and every per-question result in JSON.

The eval is the deliverable. A retrieval demo shows one good answer; this shows the rate,
the refusals, and the failures.

**Result, 2026-10-31:** baseline **25 / 30**, improved **27 / 30**, refusals **3 / 3**.
The bar was set before the run at 26 of 30 and 3 of 3. The baseline fails it, the improved
configuration clears it.

## Reproduce from a clean clone

The full eval needs the private corpus and Postgres. A reviewer can run the same
harness end to end on a published synthetic corpus, with no database:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
make demo
```

Prerequisites: Ollama running locally with `nomic-embed-text` and `qwen2.5:7b`
(`ollama pull nomic-embed-text && ollama pull qwen2.5:7b`). `make demo` embeds
the 13 synthetic documents in `data/demo_corpus/`, answers 10 questions (8
answerable, 2 that must be refused), scores recall@5, citation correctness,
completeness and refusal accuracy, writes `reports/demo_results.json`, and exits
non-zero if a metric falls below its gate in `evals/demo_gates.json`.

Expected output on this machine, 2026-11-03:

```text
13 documents -> 13 chunks
embedded 10 questions
  D01 PASS What is the base rate per mile for a dry van?
  ...
  DU2 PASS How many people work at the Halifax depot?
queries: recall@5 8/8  citation 7/8  complete 8/8  passed 7/8  refusals 2/2
demo passed every gate in evals/demo_gates.json (18.2s)
wrote reports/demo_results.json
```

The one citation miss (D04) is the same failure the full eval records for Q09:
the right answer with the wrong source number. The demo is the scaled-down
harness; the full 30-question eval and its four committed reports are in
`reports/`. It finishes in under five minutes from cold.

## What was run and what was not


**Run on this machine:**

- Postgres 16.10 with pgvector 0.8.0, built from source for this machine because Homebrew
  installs are blocked here. The `Makefile` works with any Postgres 16 + pgvector.
- `make corpus` built 57 documents (case studies, build specs, working notes), redacted for
  client and personal names.
- `make ingest` chunked, embedded and loaded **191 fixed chunks** and **236 structural chunks**.
  Embeddings from `nomic-embed-text` (768 dims) over local Ollama; answers from `qwen2.5:7b`.
- `make eval` ran 4 configurations over 33 questions each: **132 generations**.
- `make report` rendered `reports/eval.html`.
- `pytest`: **15 unit tests pass** (no database, no model).

**Not run:**

- No cloud model, no paid API. Everything is local and offline.
- The demo corpus in `data/demo_corpus/` is synthetic and published so the harness can run without the private corpus.
- The 57-document corpus is **not published**: it contains internal build specs and commercial
  notes. The repository ships the harness, the questions, the scoring rules and the reports.
  `make corpus` rebuilds it from a configured source tree.

## Results

All four configurations use the same 30 questions, the same models, temperature 0.

| Configuration | Recall@5 | Citations | Complete | **Passed** | Refusals | Bar (26) |
|---|---|---|---|---|---|---|
| Baseline: fixed 500-token chunks, vector only | 28 | 25 | 26 | **25 / 30** | 3 / 3 | not met |
| Fixed chunks, full-text hybrid | 28 | 26 | 26 | **25 / 30** | 3 / 3 | not met |
| Structural chunks, vector only | 28 | 27 | 27 | **27 / 30** | 3 / 3 | met |
| Improved: structural chunks, full-text hybrid | 28 | 27 | 27 | **27 / 30** | 3 / 3 | met |

### The one improvement that moved the number

**Structural chunking.** Sections are packed up to a 500-token budget on heading and paragraph
boundaries, and every chunk starts with its heading path (`Document > Section`). That took the
pass count from **25 to 27** and lifted citations from 25 to 27.

**Full-text hybrid retrieval did not move it.** On the fixed chunks the hybrid stayed at 25.
On the structural chunks it stayed at 27. The corpus is small and the questions share vocabulary
with the answer chunks, so the full-text list re-ranks documents the vector list already found.
Per the rule for this build, **hybrid is dropped from the default configuration**; the shipped
default is structural chunks with vector search. It is kept in the harness as a measured variant.

**Recall@5 is 28 / 30 in every configuration.** Chunking changed how much context the model got,
not what was retrieved. The two retrieval misses are the same two questions throughout.

## The eval set

`evals/questions.json`. Each answerable question records:

1. the question;
2. the expected source document(s) that hold the evidence;
3. the key fact that must appear in the answer, as an exact phrase;
4. accepted surface variants of that fact.

Three questions have no answer in the corpus. The correct response is the exact token
`NOT IN SOURCES`.

| Metric | Rule |
|---|---|
| **Recall@5** | an expected source document appears in the top five retrieved chunks |
| **Citation correctness** | a chunk the answer cites contains the key fact, and no citation number is invalid |
| **Answer completeness** | the answer text contains the key fact or a recorded variant |
| **Refusal accuracy** | an unanswerable question returns `NOT IN SOURCES` |
| **Pass** | recall and citation and completeness all hold for that question |

Fact matching normalises case, currency symbols, thousands separators and articles. The rule is
in `src/matching.py` and is unit-tested.

## Failure report

Three questions fail at the improved configuration. Full detail, including the retrieved sources
and the answer text, is in `reports/eval_improved.json` and `reports/eval.html`.

| ID | Question | Cause |
|---|---|---|
| Q09 | Which model runs the AI step in the invoice approval workflow? | the cited chunk does not hold the fact |
| Q11 | What delivery shape does the first invoice approval option use? | retrieval missed the expected source |
| Q13 | What average hourly rate did the Berlin energy-management client show? | retrieval missed the expected source |

**What the failures say.** Retrieval is the binding constraint. Where the expected source is
missing from the top five, the model refuses, which is the designed behaviour on the wrong input.
Where the answer is right and the citation is wrong, the model named a topic neighbour instead of
the source holding the fact; that is a prompt problem, and the next fix is a citation rule that
names the source of the fact rather than the topic.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # set PG_BIN, DATABASE_URL, RAG_SOURCE_ROOT

# Postgres 16 with pgvector must be installed. Then:
make db                       # initdb, start on port 55432, create schema
make corpus                   # build data/corpus/documents.jsonl from RAG_SOURCE_ROOT
make ingest                   # chunk, embed, load both indexes
make eval                     # 4 configurations, writes reports/eval_*.json
make report                   # writes reports/eval.html
make test                     # unit tests
```

Ollama must be running with the two models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve
```

`make db PG_BIN=/opt/homebrew/opt/postgresql@16/bin` on a machine with Homebrew.

## Targets

| Target | What it does |
|---|---|
| `make db` | initialise and start the local cluster, create the schema |
| `make db-stop` | stop the cluster |
| `make corpus` | rebuild the corpus JSONL from the source tree |
| `make ingest` | chunk, embed, load `docs` and `docs_baseline` |
| `make eval` | run the four configurations |
| `make report` | render `reports/eval.html` |
| `make demo` | 10-question demo on the synthetic corpus, no Postgres |
| `make test` | unit tests |

## Layout

```
src/
  corpus.py      source discovery, HTML/PDF/markdown extraction, redaction
  chunking.py    fixed-window baseline and structural chunker
  embedding.py   Ollama client with nomic task prefixes
  db.py          schema, two indexes, vector and full-text search
  retrieval.py   vector mode and reciprocal-rank-fusion hybrid mode
  answering.py   citation-required prompt, refusal token, citation parser
  matching.py    fact normalisation and matching
  evaluate.py    four configurations, per-question scoring
  report.py      readable HTML report
  ingest.py      load pipeline
evals/questions.json   30 answerable and 3 unanswerable questions
evals/demo_questions.json  10 demo questions; evals/demo_gates.json
 data/demo_corpus/      13 synthetic documents (README says so)
reports/               eval_baseline.json, eval_fixed_hybrid.json,
                       eval_structural_vector.json, eval_improved.json, eval.html
data/corpus/           documents.jsonl (private, gitignored) and manifest.json
```

## Design notes

- **Two chunkings, one question set.** The comparison is only meaningful because both indexes
  answer the same questions with the same prompt and the same model.
- **The refusal is a first-class outcome.** The prompt demands the exact token and the harness
  scores it. A model that guesses fails three questions.
- **Expected sources are document-level, not chunk-level.** A question passes recall if any
  expected document appears in the top five, including the PDF duplicate of a case study.
- **Deterministic.** Temperature 0 and a fixed seed. `nomic-embed-text` and `qwen2.5:7b` run
  locally, so a rerun reproduces the same numbers on the same corpus.

## Limits

- 30 questions is a smoke test, not a statistical claim. One question is 3.3 points.
- One corpus, one topic mix, one embedding model, one answer model. Nothing here generalises.
- No reranking model. The next attempt is a wider candidate set plus a heading-aware rerank.
- The corpus text is private, so the published numbers cannot be reproduced without it.
