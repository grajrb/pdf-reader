"""Paragraph/sentence-aware text chunking.

Strategy (documented decisions):
- Chunking happens **per page**, so every chunk carries an exactly correct
  one-based page number. Sections that span pages are split at the page
  boundary instead of being attributed to a single (possibly wrong) page.
- Split points prefer sentence boundaries, then word boundaries, and only
  fall back to hard character cuts for pathological inputs.
- Adjacent chunks overlap by ~``overlap`` characters (snapped to a word
  boundary) to preserve context between neighbours.
- Whitespace is normalized and duplicate chunks within a document are
  dropped; whitespace-only chunks are never emitted.

The strategy is intentionally simple and language-agnostic; it can be
replaced later (e.g. table/code-aware splitting) without touching callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config

_SENTENCE_END = frozenset(".!?")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NEWLINES_RE = re.compile(r"\n{3,}")
# How far back (as a fraction of chunk size) we search for a sentence end.
_SENTENCE_SEARCH_FRACTION = 0.7
_WORD_SEARCH_FRACTION = 0.5
_MAX_WORD_SNAP = 30  # chars we may skip forward to reach a word boundary


@dataclass
class Chunk:
    text: str
    page_number: int  # one-based
    chunk_index: int  # zero-based within the document


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _snap_end_backward(text: str, start: int, end: int, chunk_size: int) -> int:
    """Prefer a sentence end, then a word boundary, before a hard cut."""
    n = len(text)
    sentence_floor = max(start + 1, start + int(chunk_size * _SENTENCE_SEARCH_FRACTION))
    for i in range(end - 1, sentence_floor - 1, -1):
        if text[i] in _SENTENCE_END and i + 1 < n and text[i + 1].isspace():
            return i + 1
    word_floor = max(start + 1, start + int(chunk_size * _WORD_SEARCH_FRACTION))
    for i in range(end - 1, word_floor - 1, -1):
        if text[i].isspace():
            return i
    return end


def _snap_start_forward(text: str, start: int, limit: int) -> int:
    """Move a few characters forward to begin on a word boundary."""
    n = len(text)
    i = start
    while i < limit and i < n and not text[i].isspace():
        i += 1
    return min(i, start + _MAX_WORD_SNAP)


def split_long_text(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """Split one page's text into overlapping chunks."""
    text = _normalize(text)
    if not text:
        return []
    n = len(text)
    pieces: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _snap_end_backward(text, start, end, chunk_size)
            if end <= start:  # pathological guard
                end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = start + 1
        nxt = _snap_start_forward(text, nxt, end)
        if nxt <= start:
            nxt = start + 1
        start = nxt
    return pieces


def chunk_pages(
    pages: list,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk extracted pages (pdf_processor.PageText) into tagged chunks."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = min(overlap if overlap is not None else config.CHUNK_OVERLAP,
                  max(chunk_size - 1, 0))
    min_chars = config.MIN_CHUNK_CHARS

    chunks: list[Chunk] = []
    seen: set[str] = set()
    index = 0
    for page in pages:
        text = getattr(page, "text", "")
        page_number = getattr(page, "page_number", 0)
        for piece in split_long_text(text, chunk_size, overlap):
            if len(piece) < min_chars:
                continue
            key = piece.casefold()
            if key in seen:  # avoid duplicate chunks (e.g. repeated headers)
                continue
            seen.add(key)
            chunks.append(
                Chunk(text=piece, page_number=page_number, chunk_index=index)
            )
            index += 1
    return chunks
