"""Shared fixtures: temp databases, a deterministic offline embedder, and a
synthetic document corpus covering the grounding scenarios from the spec.

Nothing here needs Ollama: FakeEmbedder is a deterministic bag-of-words
vectorizer (stable via crc32, NOT Python's randomized hash()), so cosine
similarity approximates lexical overlap. "Override" phrases map to unit
vectors living in a reserved subspace (the last dimension), making them
exactly orthogonal to ordinary bag-of-words vectors — this lets tests
simulate paraphrase/semantic matches without a real model.
"""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import database as dbmod  # noqa: E402
from src.database import Database  # noqa: E402


class FakeEmbedder:
    """Deterministic, offline stand-in for the Ollama embedding model."""

    def __init__(self, dim: int = 48, model: str = "test-embedder", overrides=None):
        self.dim = dim
        self.model = model
        # overrides: list of (substring, vector) — matching texts get that vector
        self.overrides = overrides or []
        self.embed_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _vec(self, text: str) -> np.ndarray:
        for key, vec in self.overrides:
            if key.lower() in text.lower():
                return np.asarray(vec, dtype=np.float32)
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            v[zlib.crc32(tok.encode("utf-8")) % (self.dim - 1)] += 1.0
        norm = float(np.linalg.norm(v))
        if norm == 0:
            v[0] = 1.0
            norm = 1.0
        return v / norm

    def ensure_available(self) -> None:
        return None

    def embed_documents(self, texts, progress=None):
        self.embed_calls.append(list(texts))
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self._vec(text)


def unit(dim: int, index: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def seed_document(db: Database, filename: str, pages: dict[int, str],
                  embedder: FakeEmbedder, status: str = "indexed") -> int:
    """Insert a document + chunks + embeddings directly (bypasses PDF parsing)."""
    from src import chunker

    page_objs = [type("P", (), {"page_number": p, "text": t})() for p, t in sorted(pages.items())]
    chunks = chunker.chunk_pages(page_objs)
    vectors = (
        embedder.embed_documents([c.text for c in chunks])
        if chunks else None
    )
    text = "".join(pages.values())
    return db.replace_document(
        filename=filename,
        file_hash=filename,  # unique per test doc
        file_size=len(text),
        page_count=len(pages),
        char_count=len(text),
        chunk_count=len(chunks),
        empty_pages=[],
        status=status,
        message="",
        embedding_status="ok" if chunks else "none",
        chunks=[(c.page_number, c.chunk_index, c.text) for c in chunks],
        vectors=vectors,
        embedding_model=embedder.model if chunks else None,
    )


# The six-doc synthetic corpus required by the grounding test spec.
CORPUS: dict[str, dict[int, str]] = {
    "alpha_training.pdf": {
        1: "The Nova training plan requires exactly 3 sessions per week. "
           "Each session lasts 45 minutes and begins with a 10 minute warmup. "
           "Progressive overload is applied every second week.",
        2: "Rest day protocol: take at least two full rest days per week. "
           "Sleep at least eight hours per night for recovery.",
    },
    "beta_nutrition.pdf": {
        1: "Daily protein target is 150 grams for the Nova program. "
           "Spread protein across four meals for absorption.",
        2: "Hydration guideline: drink 2 liters of water daily, more when training.",
    },
    "conflict_a.pdf": {
        1: "The service contract renews on 2026-03-01 according to clause 4. "
           "Early termination requires 30 days notice.",
    },
    "conflict_b.pdf": {
        1: "The service contract renews on 2026-06-15 per the amended appendix. "
           "Early termination requires 60 days notice.",
    },
    "glossary.pdf": {
        1: "Sodium chloride is common table salt used for food preservation. "
           "Potassium chloride is a salt substitute.",
    },
    # "empty_scan.pdf" is added with status='no_text' and no chunks.
}


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def tmp_db(tmp_path) -> Database:
    db = Database(tmp_path / "test_knowledge.db")
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def seeded_db(tmp_db, fake_embedder) -> Database:
    for name, pages in CORPUS.items():
        seed_document(tmp_db, name, pages, fake_embedder)
    # A scanned PDF with no extractable text.
    tmp_db.replace_document(
        filename="empty_scan.pdf",
        file_hash="empty_scan.pdf",
        file_size=100,
        page_count=3,
        char_count=0,
        chunk_count=0,
        empty_pages=[1, 2, 3],
        status="no_text",
        message="No extractable text found in this PDF.",
        embedding_status="none",
    )
    return tmp_db
