"""Command-line ingestion (no UI needed).

Usage:
    python ingest.py                 # index the configured PDF dir (gym/) incrementally
    python ingest.py --rebuild       # wipe the index and re-ingest
    python ingest.py --dir some/dir  # index a different directory
"""
from __future__ import annotations

import argparse

import config
from src.database import Database
from src.embeddings import Embedder
from src.ingestion import IngestionService


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the local index.")
    parser.add_argument(
        "--dir", default=None,
        help=f"PDF directory (default: {config.PDF_DIR})",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop the index and re-ingest."
    )
    args = parser.parse_args()

    db = Database(config.DB_PATH)
    db.initialize()
    service = IngestionService(db, embedder=Embedder(), pdf_dir=args.dir)

    def progress(name: str, i: int, total: int) -> None:
        print(f"[{i}/{total}] {name}")

    report = (
        service.rebuild(args.dir, progress=progress)
        if args.rebuild
        else service.ingest_directory(args.dir, progress=progress)
    )

    print("\n%-40s %-10s %6s %6s %8s  %s" % (
        "FILE", "STATUS", "PAGES", "CHUNKS", "EMBEDDED", "MESSAGE"))
    for e in report.entries:
        print("%-40s %-10s %6d %6d %8s  %s" % (
            e.filename[:40], e.status, e.pages, e.chunks,
            "yes" if e.embedded else "no", e.message[:80]))
    print(f"\nDone: {report.indexed} indexed/updated, "
          f"{len(report.entries)} total entries. Database: {db.path}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
