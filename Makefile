PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
PYTEST  ?= .venv/bin/pytest
PG_BIN  ?=
ENV      = set -a; [ -f ./.env ] && . ./.env; set +a

.DEFAULT_GOAL := help

help:
	@echo "make db        initialise and start the local Postgres cluster, then create the schema"
	@echo "make db-stop   stop the local Postgres cluster"
	@echo "make corpus    rebuild data/corpus/documents.jsonl from the source tree"
	@echo "make ingest    chunk, embed and load both indexes"
	@echo "make eval      run the 30-question eval over four configurations"
	@echo "make report    render reports/eval.html"
	@echo "make test      run the unit tests"
	@echo "make clean     remove generated reports and the local cluster"

db:
	@$(ENV); PG_BIN="$(PG_BIN)" bash scripts/db.sh up
	@$(ENV); $(PY) -c "from src.db import Database; Database().init_schema(); print('schema ready')"

db-stop:
	@$(ENV); bash scripts/db.sh stop

corpus:
	@$(ENV); $(PY) -m src.corpus

ingest:
	@$(ENV); $(PY) -m src.ingest

eval:
	@$(ENV); $(PY) -m src.evaluate

report:
	@$(ENV); $(PY) -m src.report

test:
	@$(ENV); $(PYTEST)

clean:
	rm -f reports/eval_*.json reports/eval.html data/corpus/ingest_summary.json
	@echo "kept the corpus and the database"

.PHONY: help db db-stop corpus ingest eval report test clean
