"""PDF text extraction built on PyMuPDF (fitz).

Guarantees:
- Original PDF files are opened read-only and never modified.
- Text is extracted page by page with one-based page numbers.
- Malformed, encrypted and text-free PDFs produce structured results with a
  clear status/message instead of being silently ignored.
- Pages that contain images but (almost) no text are flagged as likely
  scanned, with an actionable OCR hint.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - env check, not a unit path
    fitz = None  # type: ignore[assignment]

# A page with fewer extractable characters than this (but some images) is
# treated as a likely scanned page.
_SCANNED_TEXT_LIMIT = 20
_READ_CHUNK = 1024 * 1024

OCR_HINT = (
    "This PDF appears to contain little or no extractable text (likely a "
    "scan or image-only document). To make it searchable, run local OCR "
    "first, e.g.:  ocrmypdf --skip-text input.pdf output.pdf  (Tesseract "
    "runs fully offline), then reindex the OCR'd file."
)


@dataclass
class PageText:
    page_number: int  # one-based
    text: str
    has_images: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class ExtractionResult:
    filename: str
    path: str
    status: str  # "ok" | "no_text" | "encrypted" | "error"
    message: str = ""
    page_count: int = 0
    pages: list[PageText] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)
    likely_scanned_pages: list[int] = field(default_factory=list)
    char_count: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def sha256_file(path: str | Path) -> str:
    """SHA-256 of the file bytes (read-only; never alters the file)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_READ_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def extract_pdf(path: str | Path) -> ExtractionResult:
    """Extract text page-by-page from a PDF, with robust failure handling."""
    path = Path(path)
    filename = path.name
    if fitz is None:
        return ExtractionResult(
            filename=filename,
            path=str(path),
            status="error",
            message="PyMuPDF is not installed. Run: pip install pymupdf",
        )
    if not path.is_file():
        return ExtractionResult(
            filename=filename,
            path=str(path),
            status="error",
            message="File does not exist.",
        )
    if path.stat().st_size == 0:
        return ExtractionResult(
            filename=filename,
            path=str(path),
            status="error",
            message="File is empty (0 bytes).",
        )

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # malformed PDFs raise several exception types
        return ExtractionResult(
            filename=filename,
            path=str(path),
            status="error",
            message=f"Could not open PDF (malformed or unsupported): {exc}",
        )

    try:
        if doc.needs_pass:
            return ExtractionResult(
                filename=filename,
                path=str(path),
                status="encrypted",
                message="PDF is encrypted/password-protected. Decrypt it "
                "locally (e.g. qpdf --password='' --decrypt in.pdf out.pdf) "
                "and re-add the decrypted copy.",
            )
        if doc.page_count == 0:
            return ExtractionResult(
                filename=filename,
                path=str(path),
                status="no_text",
                message="PDF contains zero pages.",
            )

        pages: list[PageText] = []
        empty_pages: list[int] = []
        scanned_pages: list[int] = []
        char_count = 0
        for index in range(doc.page_count):
            page = doc.load_page(index)
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            has_images = False
            try:
                has_images = bool(page.get_images(full=True))
            except Exception:
                has_images = False
            stripped = text.strip()
            if not stripped:
                empty_pages.append(index + 1)
                if has_images:
                    scanned_pages.append(index + 1)
            elif len(stripped) < _SCANNED_TEXT_LIMIT and has_images:
                scanned_pages.append(index + 1)
            char_count += len(text)
            pages.append(
                PageText(page_number=index + 1, text=text, has_images=has_images)
            )
    finally:
        doc.close()

    if char_count == 0:
        message = "No extractable text found in this PDF."
        if scanned_pages or any(p.has_images for p in pages):
            message += " " + OCR_HINT
        return ExtractionResult(
            filename=filename,
            path=str(path),
            status="no_text",
            message=message,
            page_count=len(pages),
            pages=pages,
            empty_pages=empty_pages,
            likely_scanned_pages=scanned_pages,
            char_count=0,
        )

    message = ""
    if scanned_pages:
        message = (
            f"{len(scanned_pages)} page(s) look scanned/image-only and may be "
            "missing from search results. " + OCR_HINT
        )
    return ExtractionResult(
        filename=filename,
        path=str(path),
        status="ok",
        message=message,
        page_count=len(pages),
        pages=pages,
        empty_pages=empty_pages,
        likely_scanned_pages=scanned_pages,
        char_count=char_count,
    )
