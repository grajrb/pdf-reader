"""Ingestion + extraction tests using tiny generated PDFs (needs PyMuPDF).

Covers: page-wise extraction with page numbers, empty/text-free PDFs,
encrypted and malformed files, duplicate-hash detection, change detection,
reindex/remove consistency of chunks + FTS, and ingestion reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

fitz = pytest.importorskip("pymupdf", reason="PyMuPDF not installed")

from src.database import Database  # noqa: E402
from src.ingestion import IngestionService  # noqa: E402
from src.pdf_processor import extract_pdf, sha256_file  # noqa: E402
from tests.conftest import FakeEmbedder  # noqa: E402


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    return d


@pytest.fixture
def service(pdf_dir, tmp_path):
    db = Database(tmp_path / "ingest.db")
    db.initialize()
    svc = IngestionService(
        db, embedder=FakeEmbedder(), pdf_dir=pdf_dir,
        chunk_size=400, chunk_overlap=80,
    )
    yield svc
    db.close()


def test_extract_page_by_page_with_page_numbers(pdf_dir):
    p = make_pdf(pdf_dir / "two_pages.pdf", ["Page one text apple.", "Page two text banana."])
    result = extract_pdf(p)
    assert result.ok
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "apple" in result.pages[0].text
    assert "banana" in result.pages[1].text


def test_textless_pdf_reported_not_silent(pdf_dir):
    p = make_pdf(pdf_dir / "blank.pdf", ["", "", ""])
    result = extract_pdf(p)
    assert result.status == "no_text"
    assert result.page_count == 3
    assert result.empty_pages == [1, 2, 3]
    assert "OCR" in result.message


def test_malformed_pdf_reported(pdf_dir):
    p = pdf_dir / "broken.pdf"
    p.write_bytes(b"this is not a pdf at all")
    result = extract_pdf(p)
    assert result.status == "error"
    assert "malformed" in result.message.lower() or "open" in result.message.lower()


def test_encrypted_pdf_reported(pdf_dir):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret content")
    p = pdf_dir / "locked.pdf"
    doc.save(str(p), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="hunter2")
    doc.close()
    result = extract_pdf(p)
    assert result.status == "encrypted"
    assert "password" in result.message.lower() or "encrypt" in result.message.lower()


def test_empty_file_reported(pdf_dir):
    p = pdf_dir / "zero.pdf"
    p.write_bytes(b"")
    assert extract_pdf(p).status == "error"


def test_original_pdf_never_modified(pdf_dir):
    p = make_pdf(pdf_dir / "immutable.pdf", ["Some stable content."])
    before = sha256_file(p)
    service_result_ok = extract_pdf(p)
    assert service_result_ok.ok
    assert sha256_file(p) == before


def test_ingest_directory_and_report(service, pdf_dir):
    make_pdf(pdf_dir / "a.pdf", ["Alpha document content about apples."])
    make_pdf(pdf_dir / "b.pdf", ["Beta document content about berries."])
    report = service.ingest_directory()
    assert {e.filename for e in report.entries} == {"a.pdf", "b.pdf"}
    assert all(e.status == "indexed" and e.embedded for e in report.entries)
    assert all(e.pages == 1 and e.chunks >= 1 for e in report.entries)
    docs = service.db.list_documents()
    assert len(docs) == 2
    assert all(d["embedding_status"] == "ok" for d in docs)


def test_incremental_skip_unchanged_and_detect_changes(service, pdf_dir):
    p = make_pdf(pdf_dir / "doc.pdf", ["Original text about herons."])
    first = service.ingest_file(p)
    assert first.status == "indexed"

    second = service.ingest_file(p)
    assert second.status == "unchanged"

    # Change the file's bytes -> re-ingest must update, not duplicate.
    make_pdf(pdf_dir / "doc.pdf", ["Totally new text about pelicans."])
    third = service.ingest_file(p)
    assert third.status == "updated"
    assert len(service.db.fts_search("pelicans", 10)) == 1
    assert len(service.db.fts_search("herons", 10)) == 0
    assert len(service.db.list_documents()) == 1


def test_duplicate_content_not_indexed_twice(service, pdf_dir):
    import shutil

    make_pdf(pdf_dir / "original.pdf", ["Identical body text for hashing."])
    # A true byte-for-byte copy -> same SHA-256 -> duplicate detection.
    shutil.copyfile(pdf_dir / "original.pdf", pdf_dir / "copy.pdf")
    r1 = service.ingest_file(pdf_dir / "original.pdf")
    r2 = service.ingest_file(pdf_dir / "copy.pdf")
    assert r1.status == "indexed"
    assert r2.status == "duplicate"
    indexed = [d for d in service.db.list_documents() if d["status"] == "indexed"]
    assert len(indexed) == 1


def test_remove_document_keeps_fts_consistent(service, pdf_dir):
    make_pdf(pdf_dir / "gone.pdf", ["Unique walrus information here."])
    service.ingest_file(pdf_dir / "gone.pdf")
    assert len(service.db.fts_search("walrus", 10)) == 1
    assert service.remove_document("gone.pdf") is True
    assert service.remove_document("gone.pdf") is False
    assert service.db.list_documents() == []
    assert service.db.fts_search("walrus", 10) == []
    assert service.db._c().execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_rebuild_recreates_index(service, pdf_dir):
    make_pdf(pdf_dir / "x.pdf", ["X-ray fish live in rivers."])
    make_pdf(pdf_dir / "y.pdf", ["Yaks live in mountains."])
    service.ingest_directory()
    assert len(service.db.list_documents()) == 2
    report = service.rebuild()
    assert report.indexed == 2
    assert len(service.db.list_documents()) == 2
    assert len(service.db.fts_search("yaks", 10)) == 1


def test_failed_embedding_still_indexes_text(tmp_path, pdf_dir):
    from src.embeddings import EmbeddingError

    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts, progress=None):
            raise EmbeddingError("Ollama is down")

    db = Database(tmp_path / "broken.db")
    db.initialize()
    svc = IngestionService(
        db, embedder=BrokenEmbedder(), pdf_dir=pdf_dir, chunk_size=400, chunk_overlap=80
    )
    p = make_pdf(pdf_dir / "kw.pdf", ["Keyword searchable content about zebras."])
    entry = svc.ingest_file(p)
    assert entry.status == "indexed"
    assert entry.embedded is False
    assert "keyword search works" in entry.message.lower()
    doc = db.get_document("kw.pdf")
    assert doc["embedding_status"] == "failed"
    # keyword path still functional
    assert len(db.fts_search("zebras", 10)) == 1
    # repair with a healthy embedder succeeds
    svc2 = IngestionService(
        db, embedder=FakeEmbedder(), pdf_dir=pdf_dir, chunk_size=400, chunk_overlap=80
    )
    assert svc2.repair_embeddings(doc["id"]) is True
    assert db.get_document("kw.pdf")["embedding_status"] == "ok"
    db.close()


def test_scanned_pdf_warning_surfaces_in_report(service, pdf_dir):
    # A blank page with no text -> no_text status with OCR hint in message.
    p = make_pdf(pdf_dir / "scan.pdf", [""])
    entry = service.ingest_file(p)
    assert entry.status == "no_text"
    assert "OCR" in entry.message
    doc = service.db.get_document("scan.pdf")
    assert doc["status"] == "no_text"
    assert doc["chunk_count"] == 0


def test_page_numbers_survive_ingestion(service, pdf_dir):
    import json

    p = make_pdf(pdf_dir / "multi.pdf", ["First page content apples.", "", "Third page content oranges."])
    service.ingest_file(p)
    doc = service.db.get_document("multi.pdf")
    assert doc["page_count"] == 3
    assert json.loads(doc["empty_pages"]) == [2]  # middle page empty
    rows = service.db._c().execute(
        "SELECT page_number FROM chunks WHERE document_id = ?", (doc["id"],)
    ).fetchall()
    pages = {r[0] for r in rows}
    assert pages == {1, 3}
