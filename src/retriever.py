"""Hybrid retrieval: NumPy cosine similarity + SQLite FTS5, fused with RRF.

How it works
------------
1. Semantic leg: the query is embedded with the same model used for chunks;
   cosine similarity (dot product of L2-normalized vectors) ranks all stored
   embeddings; the top ``semantic_candidates`` are kept.
2. Keyword leg: SQLite FTS5/BM25 over the same chunks; the top
   ``keyword_candidates`` are kept.
3. Fusion: Reciprocal Rank Fusion, score = sum over legs of 1/(k + rank),
   with rank starting at 1. This is score-scale-free, so BM25 and cosine
   values never need to be comparable.
4. Relevance gate: a fused result is kept only if it cleared at least one
   leg's absolute threshold (``min_cosine`` or ``min_bm25``). When nothing
   survives, the caller abstains *before* touching the LLM.

Thresholds are configurable per embedding model — see README's
"Calibrating the relevance threshold" for the calibration procedure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from src.database import Database
from src.embeddings import Embedder


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    text: str
    semantic_score: float | None = None  # cosine similarity, if in semantic leg
    keyword_score: float | None = None  # BM25-style score, if in keyword leg
    fused_score: float = 0.0


@dataclass
class RetrievalOutcome:
    query: str
    results: list[RetrievedChunk] = field(default_factory=list)
    best_semantic: float | None = None
    best_keyword: float | None = None
    semantic_considered: int = 0
    keyword_considered: int = 0

    @property
    def has_results(self) -> bool:
        return bool(self.results)


class HybridRetriever:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        top_k: int | None = None,
        semantic_candidates: int | None = None,
        keyword_candidates: int | None = None,
        rrf_k: int | None = None,
        min_cosine: float | None = None,
        min_bm25: float | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.top_k = top_k or config.DEFAULT_TOP_K
        self.semantic_candidates = (
            semantic_candidates or config.SEMANTIC_CANDIDATES
        )
        self.keyword_candidates = keyword_candidates or config.KEYWORD_CANDIDATES
        self.rrf_k = rrf_k if rrf_k is not None else config.RRF_K
        self.min_cosine = min_cosine if min_cosine is not None else config.MIN_COSINE_SIMILARITY
        self.min_bm25 = min_bm25 if min_bm25 is not None else config.MIN_BM25_SCORE

    def retrieve(
        self,
        query: str,
        doc_ids: list[int] | None = None,
        top_k: int | None = None,
    ) -> RetrievalOutcome:
        top_k = top_k or self.top_k
        outcome = RetrievalOutcome(query=query)
        query_text = (query or "").strip()
        if not query_text:
            return outcome

        # ---- semantic leg (embeddings produced by this model only) ----
        semantic: dict[int, float] = {}
        chunk_ids, matrix = self.db.load_vectors(
            self.embedder.model, doc_ids=doc_ids
        )
        outcome.semantic_considered = len(chunk_ids)
        if chunk_ids:
            q = self.embedder.embed_query(query_text)
            qn = float(np.linalg.norm(q))
            if qn > 0:
                sims = matrix @ (q / qn)
                k = min(self.semantic_candidates, len(chunk_ids))
                # argpartition for top-k, then sort those by score desc
                idx = np.argpartition(-sims, k - 1)[:k]
                order = idx[np.argsort(-sims[idx])]
                for i in order:
                    semantic[chunk_ids[int(i)]] = float(sims[int(i)])

        # ---- keyword leg (FTS5) ----
        keyword: dict[int, float] = {}
        rows = self.db.fts_search(
            query_text, self.keyword_candidates, doc_ids=doc_ids
        )
        outcome.keyword_considered = len(rows)
        for r in rows:
            keyword[r["chunk_id"]] = r["keyword_score"]

        outcome.best_semantic = max(semantic.values(), default=None)
        outcome.best_keyword = max(keyword.values(), default=None)

        # ---- Reciprocal Rank Fusion ----
        fused: dict[int, float] = {}
        for rank, chunk_id in enumerate(
            sorted(semantic, key=lambda c: -semantic[c]), start=1
        ):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, chunk_id in enumerate(
            sorted(keyword, key=lambda c: -keyword[c]), start=1
        ):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # ---- relevance gate: clear at least one absolute threshold ----
        passing = [
            cid
            for cid in fused
            if (cid in semantic and semantic[cid] >= self.min_cosine)
            or (cid in keyword and keyword[cid] >= self.min_bm25)
        ]
        passing.sort(key=lambda c: (-fused[c], -(semantic.get(c, -1.0))))
        passing = passing[:top_k]
        if not passing:
            return outcome

        evidence = self.db.fetch_evidence(passing)
        results: list[RetrievedChunk] = []
        for cid in passing:
            meta = evidence.get(cid)
            if meta is None:  # row vanished between queries; skip defensively
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page_number=meta["page_number"],
                    text=meta["text"],
                    semantic_score=semantic.get(cid),
                    keyword_score=keyword.get(cid),
                    fused_score=fused[cid],
                )
            )
        outcome.results = results
        return outcome
