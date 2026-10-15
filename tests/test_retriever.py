"""Hybrid retrieval tests using the offline FakeEmbedder.

Covers: retrieving a known relevant chunk, RRF fusion without duplicates,
metadata preservation, relevance thresholds, document filtering and
FTS query safety with hostile input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import HybridRetriever
from tests.conftest import FakeEmbedder, seed_document


def _retriever(db, embedder, **kw):
    kw.setdefault("min_cosine", 0.15)
    kw.setdefault("min_bm25", 3.0)
    return HybridRetriever(db, embedder, **kw)


def test_retrieves_known_relevant_chunk(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder)
    outcome = r.retrieve("How many training sessions per week are required?")
    assert outcome.has_results
    top = outcome.results[0]
    assert top.filename == "alpha_training.pdf"
    assert top.page_number == 1
    assert "3 sessions per week" in top.text


def test_filename_and_page_metadata_preserved(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder)
    outcome = r.retrieve("protein target daily grams")
    assert outcome.has_results
    assert outcome.results[0].filename == "beta_nutrition.pdf"
    assert outcome.results[0].page_number == 1
    assert isinstance(outcome.results[0].page_number, int)


def test_no_duplicates_when_chunk_in_both_legs(seeded_db, fake_embedder):
    # Query shares words with the chunk -> both semantic and keyword legs hit.
    r = _retriever(seeded_db, fake_embedder, top_k=10)
    outcome = r.retrieve("rest day protocol sleep recovery")
    ids = [c.chunk_id for c in outcome.results]
    assert len(ids) == len(set(ids)), "duplicate chunks in fused results"
    both = [c for c in outcome.results if c.semantic_score is not None
            and c.keyword_score is not None]
    assert both, "expected at least one chunk found by both legs"


def test_rrf_favors_chunks_found_by_both_legs(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder, top_k=5)
    outcome = r.retrieve("hydration guideline water liters daily")
    assert outcome.results
    top = outcome.results[0]
    assert top.filename == "beta_nutrition.pdf"
    assert top.page_number == 2


def test_relevance_threshold_filters_weak_chunks(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder, min_cosine=0.99, min_bm25=999.0)
    outcome = r.retrieve("training sessions per week")
    assert not outcome.has_results, "thresholds should reject everything"


def test_absent_information_yields_no_results(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder)
    outcome = r.retrieve("What is the airspeed velocity of an unladen swallow?")
    assert not outcome.has_results


def test_document_filter_restricts_results(seeded_db, fake_embedder):
    beta = seeded_db.get_document("beta_nutrition.pdf")["id"]
    alpha = seeded_db.get_document("alpha_training.pdf")["id"]
    r = _retriever(seeded_db, fake_embedder, top_k=10)
    outcome = r.retrieve("training sessions protein water", doc_ids=[beta])
    assert outcome.has_results
    assert all(c.document_id == beta for c in outcome.results)
    assert all(c.document_id != alpha for c in outcome.results)


def test_multi_document_question_retrieves_both_sources(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder, top_k=8)
    outcome = r.retrieve("training sessions per week and daily protein target")
    files = {c.filename for c in outcome.results}
    assert "alpha_training.pdf" in files
    assert "beta_nutrition.pdf" in files


def test_conflicting_documents_both_retrieved(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder, top_k=6)
    outcome = r.retrieve("service contract renewal date")
    files = {c.filename for c in outcome.results}
    assert "conflict_a.pdf" in files
    assert "conflict_b.pdf" in files


def test_terminology_difference_via_semantics(tmp_db, fake_embedder):
    """Query wording differs from the source; semantic leg should still hit."""
    seed_document(
        tmp_db, "salt.pdf",
        {1: "Sodium chloride preserves food and enhances flavor."},
        fake_embedder,
    )
    # Same vector for paraphrase on both sides = simulated semantic match.
    shared = np.zeros(fake_embedder.dim, dtype=np.float32)
    shared[0] = 1.0
    fake_embedder.overrides = [
        ("sodium chloride", shared.copy()),
        ("table salt", shared.copy()),
    ]
    r = _retriever(tmp_db, fake_embedder)
    outcome = r.retrieve("Why is table salt helpful for preserving food?")
    assert outcome.has_results
    assert outcome.results[0].filename == "salt.pdf"


def test_fts_handles_hostile_query(seeded_db, fake_embedder):
    r = _retriever(seeded_db, fake_embedder)
    outcome = r.retrieve('"; DROP TABLE chunks; -- * ( ) NEAR/')
    assert not outcome.has_results  # safe: no crash, no results
    assert seeded_db._c().execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
