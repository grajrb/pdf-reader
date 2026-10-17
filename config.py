"""Central configuration for the local PDF question-answering application.

All tunable settings live here. Key values can be overridden with environment
variables of the same name (see _env helpers) so nothing else in the project
hard-codes operational values.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# ------------------------------------------------------------------ app
APP_TITLE = _env("PDFQA_APP_TITLE", "Local PDF QA")
APP_ICON = "📄"

# ----------------------------------------------------------------- models
CHAT_MODEL = _env("PDFQA_CHAT_MODEL", "qwen3:8b")
EMBEDDING_MODEL = _env("PDFQA_EMBEDDING_MODEL", "embeddinggemma")

# ----------------------------------------------------------------- ollama
# Never point this at a remote host: the application is local-only by design.
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CONNECT_TIMEOUT = _env_float("PDFQA_OLLAMA_CONNECT_TIMEOUT", 5.0)
EMBEDDING_TIMEOUT = _env_float("PDFQA_EMBEDDING_TIMEOUT", 300.0)
CHAT_TIMEOUT = _env_float("PDFQA_CHAT_TIMEOUT", 600.0)
GENERATION_TEMPERATURE = _env_float("PDFQA_TEMPERATURE", 0.0)

# ------------------------------------------------------------------ paths
# Main data source: the gym/ folder holds the fitness program PDFs.
# Override with e.g. $env:PDFQA_PDF_DIR="D:\other\pdfs" if needed.
PDF_DIR = Path(_env("PDFQA_PDF_DIR", str(BASE_DIR / "gym")))
DATA_DIR = Path(_env("PDFQA_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = Path(_env("PDFQA_DB_PATH", str(DATA_DIR / "knowledge.db")))

# --------------------------------------------------------------- chunking
# ~1,000-1,200 char chunks with ~150-200 char overlap (project defaults).
CHUNK_SIZE = _env_int("PDFQA_CHUNK_SIZE", 1100)
CHUNK_OVERLAP = _env_int("PDFQA_CHUNK_OVERLAP", 170)
MIN_CHUNK_CHARS = _env_int("PDFQA_MIN_CHUNK_CHARS", 10)

# ------------------------------------------------------------- embeddings
EMBEDDING_BATCH_SIZE = _env_int("PDFQA_EMBED_BATCH_SIZE", 32)
# embeddinggemma is a "task-prefix" model: identical prefixes must be used
# for indexed documents and user queries.
QUERY_PREFIX = _env("PDFQA_QUERY_PREFIX", "task: search result\nquery: ")
DOCUMENT_PREFIX = _env("PDFQA_DOC_PREFIX", "task: search result\ndocument: ")

# -------------------------------------------------------------- retrieval
SEMANTIC_CANDIDATES = _env_int("PDFQA_SEMANTIC_CANDIDATES", 30)
KEYWORD_CANDIDATES = _env_int("PDFQA_KEYWORD_CANDIDATES", 30)
DEFAULT_TOP_K = _env_int("PDFQA_TOP_K", 6)
RRF_K = _env_int("PDFQA_RRF_K", 60)
# Calibrate per embedding model with a handful of known-relevant questions;
# see README "Calibrating the relevance threshold".
MIN_COSINE_SIMILARITY = _env_float("PDFQA_MIN_COSINE", 0.30)
MIN_BM25_SCORE = _env_float("PDFQA_MIN_BM25", 6.0)
MAX_EVIDENCE_CHARS = _env_int("PDFQA_MAX_EVIDENCE_CHARS", 6000)

# ------------------------------------------------------------- generation
ABSTENTION_MESSAGE = (
    "I could not find enough information in the provided PDFs."
)
JSON_RETRIES = _env_int("PDFQA_JSON_RETRIES", 1)

# ---------------------------------------------------------------- server
SERVER_ADDRESS = "127.0.0.1"  # localhost only; never expose publicly
SERVER_PORT = _env_int("PDFQA_PORT", 8501)
