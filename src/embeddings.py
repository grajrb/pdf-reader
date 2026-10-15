"""Local embedding generation via Ollama (default model: embeddinggemma).

All vectors are produced on this machine through the local Ollama HTTP API.
Documents and queries use the same model and the same task prefixes, and all
vectors are L2-normalized so cosine similarity is a plain dot product.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import config
from src.ollama_client import OllamaClient, OllamaError


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated or are inconsistent."""


class Embedder:
    """Batched, validated embedding generation against local Ollama."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        document_prefix: str | None = None,
        query_prefix: str | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.model = model or config.EMBEDDING_MODEL
        self.batch_size = max(1, batch_size or config.EMBEDDING_BATCH_SIZE)
        self.document_prefix = (
            config.DOCUMENT_PREFIX if document_prefix is None else document_prefix
        )
        self.query_prefix = config.QUERY_PREFIX if query_prefix is None else query_prefix

    # ------------------------------------------------------------ helpers
    def _check_response(self, vectors: list[list[float]], expected: int) -> np.ndarray:
        try:
            arr = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise EmbeddingError(
                f"Ollama returned malformed embeddings for '{self.model}': {exc}"
            ) from exc
        if arr.ndim != 2 or arr.shape[0] != expected:
            raise EmbeddingError(
                f"Expected {expected} embeddings from '{self.model}', "
                f"got shape {arr.shape}."
            )
        if arr.shape[1] == 0:
            raise EmbeddingError(f"Model '{self.model}' returned empty vectors.")
        if not np.isfinite(arr).all():
            raise EmbeddingError(
                f"Model '{self.model}' returned NaN/inf values."
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _embed_with_prefix(
        self,
        texts: Sequence[str],
        prefix: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed texts in batches; returns an (n, dim) normalized matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        prefixed = [prefix + t for t in texts]
        batches: list[np.ndarray] = []
        total = len(prefixed)
        done = 0
        try:
            for start in range(0, total, self.batch_size):
                batch = prefixed[start : start + self.batch_size]
                raw = self.client.embed(self.model, batch)
                arr = self._check_response(raw, len(batch))
                if batches and arr.shape[1] != batches[0].shape[1]:
                    raise EmbeddingError(
                        f"Inconsistent embedding dimension from '{self.model}': "
                        f"{batches[0].shape[1]} vs {arr.shape[1]}."
                    )
                batches.append(arr)
                done += len(batch)
                if progress is not None:
                    progress(done, total)
        except OllamaError as exc:
            raise EmbeddingError(str(exc)) from exc
        matrix = np.vstack(batches) if batches else np.empty((0, 0), dtype=np.float32)
        return matrix

    # ------------------------------------------------------------ public
    def ensure_available(self) -> None:
        """Fail fast with actionable messages when Ollama/model are missing."""
        reachable, message = self.client.ping()
        if not reachable:
            raise EmbeddingError(message)
        if not self.client.has_model(self.model):
            raise EmbeddingError(
                f"Embedding model '{self.model}' is not installed. "
                f"Download it with:  ollama pull {self.model}"
            )

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        return self._embed_with_prefix(texts, self.document_prefix, progress)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._embed_with_prefix([text], self.query_prefix)
        return vec[0]
