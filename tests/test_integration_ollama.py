"""Optional integration tests against a locally running Ollama.

Skipped by default. To run (requires `ollama pull embeddinggemma` and
`ollama pull qwen3:8b`, plus a few minutes):

    # Windows PowerShell
    $env:PDFQA_RUN_INTEGRATION="1"; python -m pytest tests/test_integration_ollama.py -v
    # macOS/Linux
    PDFQA_RUN_INTEGRATION=1 python -m pytest tests/test_integration_ollama.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.answer_generator import AnswerGenerator  # noqa: E402
from src.database import Database  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.ollama_client import OllamaClient, OllamaError  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402
from tests.conftest import CORPUS, seed_document  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("PDFQA_RUN_INTEGRATION") != "1",
    reason="Set PDFQA_RUN_INTEGRATION=1 to run tests against local Ollama models.",
)


@pytest.fixture(scope="module")
def live_check():
    client = OllamaClient()
    reachable, _ = client.ping()
    if not reachable:
        pytest.skip("Ollama is not running")
    if not client.has_model(config.EMBEDDING_MODEL):
        pytest.skip(f"{config.EMBEDDING_MODEL} not installed")
    return client


@pytest.fixture(scope="module")
def real_generator(tmp_path_factory, live_check):
    db = Database(tmp_path_factory.mktemp("it") / "it.db")
    db.initialize()
    embedder = Embedder(client=live_check, model=config.EMBEDDING_MODEL)
    for name, pages in CORPUS.items():
        seed_document(db, name, pages, embedder)
    retriever = HybridRetriever(db, embedder, top_k=6)
    gen = AnswerGenerator(retriever, client=live_check, model=config.CHAT_MODEL)
    yield gen
    db.close()


def test_real_embedding_dims_consistent(live_check):
    embedder = Embedder(client=live_check, model=config.EMBEDDING_MODEL)
    m = embedder.embed_documents(["hello world", "another sentence"])
    assert m.shape[0] == 2 and m.shape[1] > 0


def test_real_answer_is_grounded_with_citations(real_generator):
    answer = real_generator.ask(
        "How many training sessions per week does the Nova plan require?"
    )
    assert not answer.insufficient_evidence
    assert answer.citations
    assert all(c.filename.endswith(".pdf") for c in answer.citations)
    assert answer.citations[0].page >= 1


def test_real_abstention_on_absent_topic(real_generator):
    answer = real_generator.ask(
        "What is the recommended dosage of quantum flux capacitor oil?"
    )
    assert answer.insufficient_evidence
    assert answer.text == config.ABSTENTION_MESSAGE
