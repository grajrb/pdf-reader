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

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

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
