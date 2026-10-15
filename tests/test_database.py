"""Database tests: schema init, persistence, CRUD, FTS synchronization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import Database, pack_embedding, sanitize_fts_query


def _add_doc(db: Database, filename: str, chunks, model="m1", dim=8):
    vectors = None
    if chunks:
        vectors = np.stack(
            [np.full(dim, float(i + 1), dtype=np.float32) for i in range(len(chunks))]
        )
    return db.replace_document(
        filename=filename,
        file_hash=f"hash-{filename}",
        file_size=100,
        page_count=2,
        char_count=sum(len(t) for _, _, t in chunks),
        chunk_count=len(chunks),
        empty_pages=[],
        status="indexed",
        message="",
        embedding_status="ok" if chunks else "none",
        chunks=chunks,
        vectors=vectors,
        embedding_model=model if chunks else None,
    )


def test_schema_created_and_versioned(tmp_path):
    db = Database(tmp_path / "a.db")
    db.initialize()
    conn = db._c()
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"documents", "chunks", "chunks_fts", "meta"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 1
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    db.close()


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "p.db"
    db = Database(path)
    db.initialize()
    _add_doc(db, "doc.pdf", [(1, 0, "hello world alpha"), (2, 1, "second page beta")])
    db.close()

    db2 = Database(path)
    db2.initialize()
    doc = db2.get_document("doc.pdf")
    assert doc is not None
    assert doc["chunk_count"] == 2
    assert doc["page_count"] == 2
    hits = db2.fts_search("alpha", 10)
    assert len(hits) == 1
    db2.close()


def test_replace_document_replaces_old_chunks_and_fts(tmp_path):
    db = Database(tmp_path / "r.db")
    db.initialize()
    _add_doc(db, "doc.pdf", [(1, 0, "original text giraffe"), (1, 1, "more giraffe facts")])
    assert len(db.fts_search("giraffe", 10)) == 2

    _add_doc(db, "doc.pdf", [(1, 0, "completely different zebra content")])
    assert len(db.fts_search("giraffe", 10)) == 0, "stale FTS entries remain"
    assert len(db.fts_search("zebra", 10)) == 1
    assert db.get_document("doc.pdf")["chunk_count"] == 1
    total = db._c().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert total == 1, "orphaned chunks remain"
    db.close()


def test_delete_document_cascades_and_syncs_fts(tmp_path):
    db = Database(tmp_path / "d.db")
    db.initialize()
    doc_id = _add_doc(db, "keep.pdf", [(1, 0, "elephant habitat facts")])
    _add_doc(db, "gone.pdf", [(1, 0, "lion savanna facts")])
    assert len(db.fts_search("facts", 10)) == 2

    gone = db.get_document("gone.pdf")
    db.delete_document(gone["id"])

    assert db.get_document("gone.pdf") is None
    rows = db._c().execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (gone["id"],)
    ).fetchone()[0]
    assert rows == 0
    assert len(db.fts_search("savanna", 10)) == 0
    assert len(db.fts_search("elephant", 10)) == 1
    assert db.get_document("keep.pdf")["id"] == doc_id
    db.close()


def test_clear_all_removes_everything(tmp_path):
    db = Database(tmp_path / "c.db")
    db.initialize()
    _add_doc(db, "a.pdf", [(1, 0, "alpha content")])
    _add_doc(db, "b.pdf", [(1, 0, "beta content")])
    db.clear_all()
    assert db.list_documents() == []
    assert db._c().execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert db.fts_search("content", 10) == []
    db.close()


def test_fts_synced_on_chunk_text_update(tmp_path):
    db = Database(tmp_path / "u.db")
    db.initialize()
    _add_doc(db, "doc.pdf", [(1, 0, "narwhal arctic whale")])
    conn = db._c()
    with conn:
        conn.execute(
            "UPDATE chunks SET text = ? WHERE document_id = ?",
            ("walrus arctic mammal", db.get_document("doc.pdf")["id"]),
        )
    assert len(db.fts_search("narwhal", 10)) == 0
    assert len(db.fts_search("walrus", 10)) == 1
    db.close()


def test_embedding_roundtrip_and_model_filter(tmp_path):
    db = Database(tmp_path / "e.db")
    db.initialize()
    _add_doc(db, "doc.pdf", [(1, 0, "one"), (1, 1, "two")], model="model-a", dim=4)
    ids, mat = db.load_vectors("model-a")
    assert len(ids) == 2
    assert mat.shape == (2, 4)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0)
    ids_b, mat_b = db.load_vectors("model-b")
    assert ids_b == [] and mat_b.size == 0
    other = _add_doc(db, "other.pdf", [(1, 0, "three")], model="model-a", dim=4)
    ids_f, _ = db.load_vectors("model-a", doc_ids=[other])
    assert len(ids_f) == 1
    db.close()


def test_chunks_needing_embeddings_and_update(tmp_path):
    db = Database(tmp_path / "n.db")
    db.initialize()
    _add_doc(db, "doc.pdf", [(1, 0, "one"), (1, 1, "two")], model="old-model")
    doc_id = db.get_document("doc.pdf")["id"]
    pending = db.chunks_needing_embeddings(doc_id, "new-model")
    assert len(pending) == 2
    vecs = np.ones((2, 4), dtype=np.float32)
    db.update_chunk_embeddings([p["id"] for p in pending], vecs, "new-model")
    assert db.chunks_needing_embeddings(doc_id, "new-model") == []
    assert len(db.load_vectors("new-model")[0]) == 2
    db.close()


def test_meta_roundtrip(tmp_path):
    db = Database(tmp_path / "m.db")
    db.initialize()
    assert db.get_meta("missing") is None
    db.set_meta("k", "v1")
    db.set_meta("k", "v2")
    assert db.get_meta("k") == "v2"
    db.close()


def test_fts_query_safety():
    assert sanitize_fts_query("hello world") == '"hello" OR "world"'
    sanitized = sanitize_fts_query("x'; DROP TABLE chunks;--")
    # FTS5 metacharacters (quotes, semicolons, operators) are stripped;
    # leftover words become inert quoted tokens.
    assert ";" not in sanitized and "'" not in sanitized and "-" not in sanitized
    assert sanitize_fts_query("") == ""
    assert sanitize_fts_query("   ???  ") == ""


def test_parameterized_queries_reject_injection(tmp_path):
    db = Database(tmp_path / "s.db")
    db.initialize()
    _add_doc(db, "doc.pdf", [(1, 0, "safety pin content")])
    assert db.get_document("doc.pdf'; DROP TABLE chunks;--") is None
    assert isinstance(db.fts_search("pin'; DROP TABLE chunks;--", 10), list)
    names = {
        r[0]
        for r in db._c().execute("SELECT name FROM sqlite_master").fetchall()
    }
    assert "chunks" in names
    db.close()


def test_pack_embedding_is_float32_blob():
    blob = pack_embedding([1.0, 2.0, 3.0])
    assert len(blob) == 3 * 4
    arr = np.frombuffer(blob, dtype=np.float32)
    assert np.allclose(arr, [1.0, 2.0, 3.0])
