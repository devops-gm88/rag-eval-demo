"""Postgres + pgvector storage for the two chunkings.

`docs`           structural chunks (the improved index)
`docs_baseline`  fixed 500-token chunks (the baseline index)

Both tables have the schema given in the session brief. The retrieval mode is a
query choice, so four configurations come from two tables and two modes.
"""
from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL, EMBED_DIM

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
    id          bigserial primary key,
    source      text,
    chunk_index int,
    text        text,
    embedding   vector({dim}),
    meta        jsonb,
    unique(source, chunk_index)
);

CREATE TABLE IF NOT EXISTS docs_baseline (LIKE docs INCLUDING ALL);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id         uuid primary key,
    status     text not null,
    sources    jsonb not null,
    chunks     int default 0,
    error      text,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);
""".format(dim=EMBED_DIM)

INDEXES = """
CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON docs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS docs_baseline_embedding_idx
    ON docs_baseline USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS docs_fts_idx
    ON docs USING gin (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS docs_baseline_fts_idx
    ON docs_baseline USING gin (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS docs_source_idx ON docs (source);
CREATE INDEX IF NOT EXISTS docs_baseline_source_idx ON docs_baseline (source);
"""

TABLES = {"fixed": "docs_baseline", "structural": "docs"}


def table_for(chunking: str) -> str:
    if chunking not in TABLES:
        raise ValueError(f"unknown chunking {chunking!r}")
    return TABLES[chunking]


def vector_literal(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


class Database:
    def __init__(self, dsn: str = DATABASE_URL):
        self.dsn = dsn
        self.conn: psycopg.Connection | None = None

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> "Database":
        if self.conn is None or self.conn.closed:
            self.conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self

    def close(self) -> None:
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ schema
    def init_schema(self, with_indexes: bool = True) -> None:
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
            if with_indexes:
                cur.execute(INDEXES)

    def reset(self, chunking: str | None = None) -> None:
        self.connect()
        tables = [table_for(chunking)] if chunking else list(TABLES.values())
        with self.conn.cursor() as cur:
            for table in tables:
                cur.execute(f"TRUNCATE {table} RESTART IDENTITY")

    def count(self, chunking: str) -> int:
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) AS n FROM {table_for(chunking)}")
            return int(cur.fetchone()["n"])

    def source_list(self, chunking: str) -> list[str]:
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT source FROM {table_for(chunking)} ORDER BY source")
            return [row["source"] for row in cur.fetchall()]

    # ------------------------------------------------------------ writes
    def insert_chunks(self, chunking: str, rows: list[dict]) -> int:
        """rows: {source, chunk_index, text, embedding, meta}"""
        self.connect()
        table = table_for(chunking)
        payload = [
            (
                row["source"],
                row["chunk_index"],
                row["text"],
                vector_literal(row["embedding"]),
                json.dumps(row.get("meta", {})),
            )
            for row in rows
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {table} (source, chunk_index, text, embedding, meta)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                    ON CONFLICT (source, chunk_index) DO UPDATE
                    SET text = excluded.text,
                        embedding = excluded.embedding,
                        meta = excluded.meta""",
                payload,
            )
        return len(payload)

    # ------------------------------------------------------------ reads
    def vector_search(self, chunking: str, query_vector: list[float], k: int) -> list[dict]:
        table = table_for(chunking)
        sql = f"""
            SELECT id, source, chunk_index, text, meta,
                   1 - (embedding <=> %s::vector) AS score
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        literal = vector_literal(query_vector)
        with self.conn.cursor() as cur:
            cur.execute(sql, (literal, literal, k))
            return cur.fetchall()

    def fts_search(self, chunking: str, query: str, k: int) -> list[dict]:
        table = table_for(chunking)
        sql = f"""
            SELECT id, source, chunk_index, text, meta,
                   ts_rank_cd(to_tsvector('english', text),
                              websearch_to_tsquery('english', %s)) AS score
            FROM {table}
            WHERE to_tsvector('english', text) @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC, id
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (query, query, k))
            return cur.fetchall()

    # ------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        try:
            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                version = cur.fetchone()
            return {
                "reachable": True,
                "pgvector": version["extversion"] if version else None,
                "chunks_structural": self.count("structural"),
                "chunks_fixed": self.count("fixed"),
            }
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return {"reachable": False, "error": str(exc)}
