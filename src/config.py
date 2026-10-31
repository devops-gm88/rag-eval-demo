"""Central configuration. Every path and model name lives here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
EVALS_DIR = ROOT / "evals"
REPORTS_DIR = ROOT / "reports"

# Default connection string for the workspace-local Postgres cluster.
# Override with DATABASE_URL for any other server.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://127.0.0.1:55432/rag_eval",
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:7b")
EMBED_DIM = 768

# Chunking
BASELINE_CHUNK_TOKENS = 500
BASELINE_OVERLAP_TOKENS = 50
STRUCTURAL_MAX_TOKENS = 500
CHARS_PER_TOKEN = 4  # estimator; nomic-embed-text and qwen both use BPE

TOP_K = 5
HYBRID_CANDIDATES = 20
RRF_K = 60

# Generation
TEMPERATURE = 0.0
NUM_CTX = 2048
REQUEST_TIMEOUT = 180
KEEP_ALIVE = "30m"
