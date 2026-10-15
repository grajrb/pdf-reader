"""Incremental ingestion pipeline: PDFs -> text -> chunks -> embeddings.

Document lifecycle decisions (documented in README):
- Every file's SHA-256 is stored; unchanged files are skipped on re-runs.
- If a file's bytes changed, its old chunks/FTS/embeddings are replaced
  atomically by ``Database.replace_document``.
- A different filename with bytes identical to an already-indexed file is
  recorded with status "duplicate" and not indexed twice.
- Failures (malformed/encrypted/empty PDFs, embedding errors) are recorded
  in the ingestion report and the document row; they never abort the batch.
- If Ollama is unavailable, text + FTS are still indexed
  (embedding_status="failed") so keyword search works immediately and
  embeddings can be repaired later with the same command.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import config
from src import chunker
from src.database import Database
from src.embeddings import Embedder, EmbeddingError
from src.pdf_processor import extract_pdf, sha256_file


@dataclass
class IngestionReportEntry:
    filename: str
    status: str  # indexed|unchanged|updated|duplicate|no_text|encrypted|error
    pages: int = 0
    chars: int = 0
    chunks: int = 0
    embedded: bool = False
    message: str = ""


@dataclass
class IngestionReport:
    entries: list[IngestionReportEntry] = field(default_factory=list)

    def add(self, entry: IngestionReportEntry) -> IngestionReportEntry:
        self.entries.append(entry)
        return entry

    @property
    def indexed(self) -> int:
        return sum(1 for e in self.entries if e.status in ("indexed", "updated"))


class IngestionService:
    def __init__(
        self,
        db: Database,
        embedder: Embedder | None = None,
        pdf_dir: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder or Embedder()
        self.pdf_dir = Path(pdf_dir if pdf_dir is not None else config.PDF_DIR)
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    # ------------------------------------------------------------ helpers
    @staticmethod
    def list_pdfs(directory: str | Path) -> list[Path]:
        d = Path(directory)
        if not d.is_dir():
            return []
        return sorted(
            (p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
            key=lambda p: p.name.lower(),
        )

    def _embed_chunks(
        self,
        chunks: list[chunker.Chunk],
        doc_id_for_status: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray | None, str]:
        """Embed chunk texts. Returns (vectors_or_None, status)."""
        try:
            vectors = self.embedder.embed_documents(
                [c.text for c in chunks], progress=progress
            )
        except EmbeddingError as exc:
            if doc_id_for_status is not None:
                self.db.set_embedding_status(doc_id_for_status, "failed")
            return None, f"failed: {exc}"
        return vectors, "ok"

    def repair_embeddings(
        self,
        doc_id: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> bool:
        """(Re)embed any chunks of a document that lack a current-model vector."""
        pending = self.db.chunks_needing_embeddings(doc_id, self.embedder.model)
        if not pending:
            self.db.set_embedding_status(doc_id, "ok")
            return True
        try:
            vectors = self.embedder.embed_documents(
                [p["text"] for p in pending], progress=progress
            )
        except EmbeddingError as exc:
            self.db.set_embedding_status(doc_id, "failed")
            self.db.set_meta("last_embedding_error", str(exc))
            return False
        self.db.update_chunk_embeddings(
            [p["id"] for p in pending], vectors, self.embedder.model
        )
        self.db.set_embedding_status(doc_id, "ok")
        return True


    # ------------------------------------------------------------ ingest
    def ingest_file(
        self,
        path: str | Path,
        force: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> IngestionReportEntry:
        """Index one PDF incrementally. Never raises for per-file problems."""
        path = Path(path)
        filename = path.name
        if not path.is_file():
            return IngestionReportEntry(
                filename=filename, status="error", message="File not found."
            )

        file_hash = sha256_file(path)
        file_size = path.stat().st_size
        existing = self.db.get_document(filename)

        # Duplicate-content detection (same bytes under a different name).
        twin = self.db.get_document_by_hash(file_hash)
        if twin is not None and twin["filename"] != filename:
            return IngestionReportEntry(
                filename=filename,
                status="duplicate",
                message=f"Identical content already indexed as "
                f"'{twin['filename']}'; skipped.",
            )

        # Unchanged file: skip extraction/embedding entirely.
        if (
            not force
            and existing is not None
            and existing["file_hash"] == file_hash
            and existing["status"] == "indexed"
        ):
            if existing["embedding_status"] != "ok":
                self.repair_embeddings(existing["id"], progress=progress)
                existing = self.db.get_document(filename) or existing
            embedded_ok = existing["embedding_status"] == "ok"
            return IngestionReportEntry(
                filename=filename,
                status="unchanged" if embedded_ok else "error",
                pages=existing["page_count"],
                chars=existing["char_count"],
                chunks=existing["chunk_count"],
                embedded=embedded_ok,
                message="Already indexed; skipped."
                if embedded_ok
                else existing["message"] or "Embedding repair failed.",
            )

        prior_status = "updated" if existing is not None else "indexed"
        extraction = extract_pdf(path)

        if extraction.status != "ok":
            doc_status = {
                "no_text": "no_text",
                "encrypted": "encrypted",
            }.get(extraction.status, "error")
            self.db.replace_document(
                filename=filename,
                file_hash=file_hash,
                file_size=file_size,
                page_count=extraction.page_count,
                char_count=extraction.char_count,
                chunk_count=0,
                empty_pages=extraction.empty_pages,
                status=doc_status,
                message=extraction.message,
                embedding_status="none",
            )
            return IngestionReportEntry(
                filename=filename,
                status=doc_status,
                pages=extraction.page_count,
                chars=extraction.char_count,
                chunks=0,
                embedded=False,

        chunks = chunker.chunk_pages(
            extraction.pages, self.chunk_size, self.chunk_overlap
        )
        if not chunks:
            message = "PDF has text on no usable pages after cleaning."
            self.db.replace_document(
                filename=filename,
                file_hash=file_hash,
                file_size=file_size,
                page_count=extraction.page_count,
                char_count=extraction.char_count,
                chunk_count=0,
                empty_pages=extraction.empty_pages,
                status="no_text",
                message=message,
                embedding_status="none",
            )
            return IngestionReportEntry(
                filename=filename,
                status="no_text",
                pages=extraction.page_count,
                chars=extraction.char_count,
                chunks=0,
                message=message,
            )

        doc_id = self.db.replace_document(
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            page_count=extraction.page_count,
            char_count=extraction.char_count,
            chunk_count=len(chunks),
            empty_pages=extraction.empty_pages,
            status="indexed",
            message=extraction.message,
            embedding_status="pending",
            chunks=[(c.page_number, c.chunk_index, c.text) for c in chunks],
        )
        vectors, embed_status = self._embed_chunks(
            chunks, doc_id_for_status=doc_id, progress=progress
        )
        if vectors is not None:
            rows = self.db._c().execute(
                "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
            self.db.update_chunk_embeddings(
                [r[0] for r in rows], vectors, self.embedder.model
            )
            self.db.set_embedding_status(doc_id, "ok")
            self.db.set_meta("embedding_model", self.embedder.model)
        else:
            self.db.set_meta("last_embedding_error", embed_status)

        message = extraction.message
        if vectors is None:
            extra = (
                "Text indexed, but embeddings failed ("
                + embed_status.removeprefix("failed: ")
                + "). Keyword search works; run index again after fixing "
                "Ollama to add semantic search."
            )
            message = f"{message} {extra}".strip()
        return IngestionReportEntry(
            filename=filename,
            status=prior_status,
            pages=extraction.page_count,
            chars=extraction.char_count,
            chunks=len(chunks),
            embedded=vectors is not None,
            message=message,
        )

    def ingest_directory(
        self,
        directory: str | Path | None = None,
        force: bool = False,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> IngestionReport:
        """Index every PDF in a directory, skipping unchanged files."""
        report = IngestionReport()
        pdfs = self.list_pdfs(directory or self.pdf_dir)
        total = len(pdfs)
        for i, path in enumerate(pdfs, start=1):
            if progress is not None:
                progress(path.name, i, total)

            def file_progress(done: int, count: int, _name: str = path.name) -> None:
                if progress is not None:
                    progress(_name, i, total)

            report.add(self.ingest_file(path, force=force, progress=file_progress))
        return report

    def remove_document(self, filename: str) -> bool:
        """Delete a document and all derived rows (chunks, FTS, embeddings)."""
        doc = self.db.get_document(filename)
        if doc is None:
            return False
        self.db.delete_document(doc["id"])
        return True

    def rebuild(
        self,
        directory: str | Path | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> IngestionReport:
        """Drop the whole index and re-ingest a directory from scratch."""
        self.db.clear_all()
        return self.ingest_directory(directory, force=True, progress=progress)

                message=extraction.message,
            )
