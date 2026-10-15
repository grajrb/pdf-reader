"""SQLite storage layer for documents, chunks, embeddings and the FTS5 index.

Design notes:
- One local database file (default ``data/knowledge.db``); no server process.
- Foreign keys are enforced; deleting a document removes its chunks, and the
  FTS index is kept in sync through AFTER INSERT/DELETE/UPDATE triggers on the
  chunks table (external-content FTS5 table, so text is stored only once).
- Embeddings are stored as little-endian float32 BLOBs (compact, fast to load
  with NumPy ``np.frombuffer``).
- All SQL uses parameterized queries; multi-row work is batched into single
  transactions (no per-row commits).
- Schema changes are tracked with ``PRAGMA user_version``.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import config

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT    NOT NULL UNIQUE,
    file_hash        TEXT    NOT NULL,
    file_size        INTEGER NOT NULL DEFAULT 0,
    page_count       INTEGER NOT NULL DEFAULT 0,
    char_count       INTEGER NOT NULL DEFAULT 0,
    chunk_count      INTEGER NOT NULL DEFAULT 0,
    empty_pages      TEXT    NOT NULL DEFAULT '[]',
    status           TEXT    NOT NULL DEFAULT 'pending',
    embedding_status TEXT    NOT NULL DEFAULT 'none',
    message          TEXT    NOT NULL DEFAULT '',
    ingested_at      TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (file_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    embedding       BLOB,
    embedding_model TEXT,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts (rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MAX_FTS_TOKENS = 32


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_fts_query(query: str) -> str:
    """Convert raw user text into a safe FTS5 OR-query of quoted tokens."""
    tokens = _FTS_TOKEN_RE.findall(query or "")[:_MAX_FTS_TOKENS]
    return " OR ".join(f'"{t}"' for t in tokens)


def pack_embedding(vector: Sequence[float]) -> bytes:
    """Encode a vector as a little-endian float32 BLOB."""
    return np.asarray(vector, dtype=np.float32).tobytes()


class Database:
    """Thin wrapper around a local SQLite connection."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path if path is not None else config.DB_PATH)
        self.conn: sqlite3.Connection | None = None

    # ---------------------------------------------------------- lifecycle
    def connect(self) -> sqlite3.Connection:
        if self.conn is not None:
            return self.conn
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        self.conn = conn
        return conn

    def initialize(self) -> None:
        """Create tables/indexes if needed and run idempotent migrations.

        All DDL uses IF NOT EXISTS so it is safe on every start; user_version
        tracks future schema changes.
        """
        conn = self.connect()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.executescript(_SCHEMA_SQL)
        if version < SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "Database":
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _c(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("Database is not connected; call connect()/initialize().")
        return self.conn

    # --------------------------------------------------------------- meta
    def get_meta(self, key: str) -> str | None:
        row = self._c().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        conn = self._c()
        with conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ---------------------------------------------------------- documents
    def get_document(self, filename: str) -> dict | None:
        row = self._c().execute(
            "SELECT * FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        return dict(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> dict | None:
        row = self._c().execute(
            "SELECT * FROM documents WHERE file_hash = ? ORDER BY id LIMIT 1",
            (file_hash,),
        ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        rows = self._c().execute(
            "SELECT * FROM documents ORDER BY filename COLLATE NOCASE"
        ).fetchall()
        return [dict(r) for r in rows]

    def replace_document(
        self,
        *,
        filename: str,
        file_hash: str,
        file_size: int,
        page_count: int,
        char_count: int,
        chunk_count: int,
        empty_pages: Sequence[int],
        status: str,
        message: str,
        embedding_status: str,
        chunks: Sequence[tuple[int, int, str]] | None = None,
        vectors: np.ndarray | None = None,
        embedding_model: str | None = None,
    ) -> int:
        """Atomically replace a document row and all of its chunks.

        ``chunks`` is a sequence of ``(page_number, chunk_index, text)``;
        ``vectors`` (optional) is an ``(n, dim)`` float array aligned with it.
        Deleting the old row cascades to its chunks and the FTS triggers keep
        the full-text index synchronized. Returns the new document id.
        """
        conn = self._c()
        now = _utcnow()
        chunks = list(chunks or [])
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError(
                f"vector count {len(vectors)} != chunk count {len(chunks)}"
            )
        with conn:
            conn.execute(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE filename = ?)",
                (filename,),
            )
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            cur = conn.execute(
                "INSERT INTO documents (filename, file_hash, file_size, page_count,"
                " char_count, chunk_count, empty_pages, status, embedding_status,"
                " message, ingested_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    filename,
                    file_hash,
                    file_size,
                    page_count,
                    char_count,
                    chunk_count,
                    json.dumps(sorted(empty_pages or [])),
                    status,
                    embedding_status,
                    message,
                    now,
                    now,
                ),
            )
            doc_id = int(cur.lastrowid)
            if chunks:
                conn.executemany(
                    "INSERT INTO chunks (document_id, page_number, chunk_index,"
                    " text) VALUES (?, ?, ?, ?)",
                    [(doc_id, p, i, t) for p, i, t in chunks],
                )
                if vectors is not None and embedding_model:
                    ids = [
                        r[0]
                        for r in conn.execute(
                            "SELECT id FROM chunks WHERE document_id = ?"
                            " ORDER BY chunk_index",
                            (doc_id,),
                        )
                    ]
                    conn.executemany(
                        "UPDATE chunks SET embedding = ?, embedding_model = ?"
                        " WHERE id = ?",
                        [
                            (pack_embedding(v), embedding_model, cid)
                            for cid, v in zip(ids, vectors)
                        ],
                    )
        return doc_id

    # ---------------------------------------------------------- embeddings
    def chunks_needing_embeddings(
        self, doc_id: int, model: str
    ) -> list[dict]:
        """Chunks of a document that are unembedded or embedded with another model."""
        rows = self._c().execute(
            "SELECT id, page_number, chunk_index, text FROM chunks"
            " WHERE document_id = ? AND (embedding IS NULL OR embedding_model IS ?)"
            " ORDER BY chunk_index",
            (doc_id, model),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_chunk_embeddings(
        self,
        chunk_ids: Sequence[int],
        vectors: np.ndarray,
        model: str,
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError(
                f"chunk id count {len(chunk_ids)} != vector count {len(vectors)}"
            )
        conn = self._c()
        with conn:
            conn.executemany(
                "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                [
                    (pack_embedding(v), model, cid)
                    for cid, v in zip(chunk_ids, vectors)
                ],
            )

    def load_vectors(
        self, model: str, doc_ids: Iterable[int] | None = None
    ) -> tuple[list[int], np.ndarray]:
        """Load normalized float32 embeddings for semantic search.

        Returns (chunk_ids, matrix) where matrix is (n, dim). Only embeddings
        produced by ``model`` are considered, so switching models invalidates
        old vectors instead of silently mixing vector spaces.
        """
        sql = (
            "SELECT id, embedding FROM chunks"
            " WHERE embedding IS NOT NULL AND embedding_model = ?"
        )
        params: list[object] = [model]
        if doc_ids is not None:
            ids = list(doc_ids)
            if not ids:
                return [], np.empty((0, 0), dtype=np.float32)
            sql += f" AND document_id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        rows = self._c().execute(sql, params).fetchall()
        chunk_ids = [r["id"] for r in rows]
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        mat = np.vstack(
            [
                np.frombuffer(r["embedding"], dtype=np.float32).astype(np.float32)
                for r in rows
            ]
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return chunk_ids, mat / norms

    # ------------------------------------------------------------- search
    def fts_search(
        self,
        query: str,
        limit: int,
        doc_ids: Sequence[int] | None = None,
    ) -> list[dict]:
        """Keyword search via FTS5/BM25. Higher ``keyword_score`` is better."""
        match = sanitize_fts_query(query)
        if not match:
            return []
        sql = (
            "SELECT c.id, c.document_id, c.page_number, -bm25(chunks_fts) AS score"
            " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
            " WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [match]
        if doc_ids:
            sql += f" AND c.document_id IN ({','.join('?' * len(doc_ids))})"
            params.extend(doc_ids)
        sql += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        rows = self._c().execute(sql, params).fetchall()
        return [
            {
                "chunk_id": r["id"],
                "document_id": r["document_id"],
                "page_number": r["page_number"],
                "keyword_score": float(r["score"]),
            }
            for r in rows
        ]

    def fetch_evidence(self, chunk_ids: Sequence[int]) -> dict[int, dict]:
        """Fetch text + filename + page for specific chunks, keyed by chunk id."""
        if not chunk_ids:
            return {}
        sql = (
            "SELECT c.id, c.document_id, c.page_number, c.text, d.filename"
            " FROM chunks c JOIN documents d ON d.id = c.document_id"
            f" WHERE c.id IN ({','.join('?' * len(chunk_ids))})"
        )
        rows = self._c().execute(sql, list(chunk_ids)).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def set_embedding_status(self, doc_id: int, status: str) -> None:
        conn = self._c()
        with conn:
            conn.execute(
                "UPDATE documents SET embedding_status = ?, updated_at = ?"
                " WHERE id = ?",
                (status, _utcnow(), doc_id),
            )

    def delete_document(self, doc_id: int) -> None:
        """Delete a document and everything derived from it (chunks + FTS)."""
        conn = self._c()
        with conn:
            # Explicit chunk delete (triggers sync FTS); cascade covers stragglers.
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def clear_all(self) -> None:
        conn = self._c()
        with conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")

        with conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
