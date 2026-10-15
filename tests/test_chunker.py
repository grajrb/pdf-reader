"""Chunking tests: size limits, overlap, boundaries, dedup, page attribution."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import chunker


class FakePage:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text


def _long_text(paragraphs: int = 12) -> str:
    return "\n\n".join(
        f"Paragraph {i} discusses topic {i} in complete sentences. "
        f"It elaborates on the subject with additional detail number {i}. "
        f"Another sentence adds context for paragraph {i} here."
        for i in range(paragraphs)
    )


def test_respects_max_chunk_size():
    pages = [FakePage(1, _long_text())]
    chunks = chunker.chunk_pages(pages, chunk_size=300, overlap=60)
    assert chunks
    assert all(len(c.text) <= 300 for c in chunks)


def test_overlap_between_adjacent_chunks():
    pages = [FakePage(1, _long_text())]
    overlap = 60
    chunks = chunker.chunk_pages(pages, chunk_size=300, overlap=overlap)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # Adjacent chunks must share a meaningful tail/head (word snapping can
        # shave a few chars off the nominal overlap).
        shared = 0
        for size in range(overlap, 0, -1):
            if prev.text.endswith(nxt.text[:size]) or nxt.text.startswith(prev.text[-size:]):
                shared = size
                break
        assert shared >= overlap - 35


def test_no_empty_or_duplicate_chunks():
    pages = [FakePage(1, _long_text() + "\n\n" + _long_text(2))]
    chunks = chunker.chunk_pages(pages, chunk_size=250, overlap=50)
    texts = [c.text for c in chunks]
    assert all(t.strip() for t in texts)
    assert len(set(texts)) == len(texts)


def test_sentence_boundaries_preferred():
    text = ("First sentence goes here and is fairly long. "
            "Second sentence continues the thought. "
            "Third sentence closes it.") * 20
    pages = [FakePage(1, text)]
    chunks = chunker.chunk_pages(pages, chunk_size=300, overlap=50)
    for c in chunks:
        # No chunk should end mid-word: last char is punctuation or space join.
        assert c.text[-1].isalnum() or c.text[-1] in ".!?"
        # And should not start mid-word (snap-forward guarantee).
        assert c.text[0].isalnum()


def test_page_metadata_preserved():
    pages = [FakePage(1, _long_text(4)), FakePage(2, _long_text(4)),
             FakePage(3, "A short third page. " * 30)]
    chunks = chunker.chunk_pages(pages, chunk_size=250, overlap=50)
    by_page = {}
    for c in chunks:
        by_page.setdefault(c.page_number, []).append(c)
    assert set(by_page) == {1, 2, 3}
    # chunk_index is unique and sequential across the document
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_and_whitespace_pages_yield_nothing():
    pages = [FakePage(1, ""), FakePage(2, "   \n\t  ")]
    assert chunker.chunk_pages(pages) == []


def test_long_single_paragraph_is_split():
    text = "word " * 2000  # one enormous paragraph, no sentence ends
    pages = [FakePage(1, text)]
    chunks = chunker.chunk_pages(pages, chunk_size=400, overlap=80)
    assert len(chunks) >= 4
    assert all(len(c.text) <= 400 for c in chunks)


def test_default_config_values_used():
    import config
    pages = [FakePage(1, _long_text(20))]
    chunks = chunker.chunk_pages(pages)
    assert all(len(c.text) <= config.CHUNK_SIZE for c in chunks)
