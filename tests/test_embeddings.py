"""Tests for embedding generation and Ollama failure handling.

All HTTP is monkeypatched: no real Ollama server or network is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings import Embedder, EmbeddingError  # noqa: E402
from src.ollama_client import (  # noqa: E402
    OllamaClient,
    OllamaModelMissing,
    OllamaUnavailable,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("" if payload is None else str(payload))

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_unavailable_server_raises_friendly_error(monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    client = OllamaClient(host="http://127.0.0.1:9")
    with pytest.raises(OllamaUnavailable) as exc:
        client.list_models()
    assert "ollama serve" in str(exc.value) or "Ollama" in str(exc.value)


def test_missing_model_instructs_pull(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: FakeResponse(payload={"models": [{"name": "llama3.2:latest"}]}),
    )
    client = OllamaClient()
    assert client.has_model("embeddinggemma") is False
    assert client.has_model("llama3.2") is True
    assert client.has_model("llama3.2:latest") is True

    embedder = Embedder(client=client, model="embeddinggemma")
    with pytest.raises(EmbeddingError) as exc:
        embedder.ensure_available()
    assert "ollama pull embeddinggemma" in str(exc.value)


def test_embed_http_404_maps_to_model_missing(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **kw: FakeResponse(404, {"error": "model 'x' not found"}),
    )
    client = OllamaClient()
    with pytest.raises(OllamaModelMissing):
        client.embed("x", ["hello"])


def test_embedding_batches_and_normalization(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        n = len(json["input"])
        # 4-dim vectors with varying norms to test normalization
        return FakeResponse(payload={
            "embeddings": [[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)]
        })

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: FakeResponse(payload={"models": [{"name": "m:latest"}]}),
    )
    embedder = Embedder(
        client=OllamaClient(), model="m:latest", batch_size=32
    )
    texts = [f"text {i}" for i in range(70)]
    matrix = embedder.embed_documents(texts)
    assert matrix.shape == (70, 4)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert len(calls) == 3, f"expected ceil(70/32)=3 batches, got {len(calls)}"


def test_query_and_document_prefixes_used(monkeypatch):
    seen = []

    def fake_post(url, json=None, timeout=None):
        seen.extend(json["input"])
        return FakeResponse(payload={"embeddings": [[1.0, 0.0]] * len(json["input"])})

    monkeypatch.setattr(requests, "post", fake_post)
    embedder = Embedder(client=OllamaClient(), model="m")
    embedder.embed_documents(["doc text"])
    embedder.embed_query("query text")
    import config
    assert seen[0] == config.DOCUMENT_PREFIX + "doc text"
    assert seen[1] == config.QUERY_PREFIX + "query text"


def test_dimension_mismatch_raises(monkeypatch):
    responses = [
        FakeResponse(payload={"embeddings": [[1.0, 0.0]]}),
        FakeResponse(payload={"embeddings": [[1.0, 0.0, 0.0]]}),
    ]

    def fake_post(url, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(requests, "post", fake_post)
    embedder = Embedder(client=OllamaClient(), model="m", batch_size=1)
    with pytest.raises(EmbeddingError) as exc:
        embedder.embed_documents(["a", "b"])
    assert "dimension" in str(exc.value).lower()


def test_nan_vectors_rejected(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **kw: FakeResponse(payload={"embeddings": [[float("nan"), 1.0]]}),
    )
    embedder = Embedder(client=OllamaClient(), model="m")
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])
