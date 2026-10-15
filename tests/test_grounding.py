"""Grounding tests for AnswerGenerator with a mocked Ollama chat client.

Covers: abstention without calling the LLM, citation validation (no invented
citations), controlled fallback for invalid JSON, model-declared insufficient
evidence, and the guarantee that chat history is never used as evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.answer_generator import AnswerGenerator, validate_model_output  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402


class FakeChatClient:
    """Scripted chat client; records every payload it is asked to send."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, model, messages, temperature=0.0, timeout=None):
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("FakeChatClient ran out of scripted responses")
        return self.responses.pop(0)


def _generator(db, embedder, responses):
    retriever = HybridRetriever(
        db, embedder, min_cosine=0.25, min_bm25=2.0, top_k=6
    )
    client = FakeChatClient(responses)
    return AnswerGenerator(retriever, client=client), client


def test_answer_with_valid_json_and_citations(seeded_db, fake_embedder):
    payload = json.dumps({
        "answer": "The Nova plan requires 3 sessions per week "
                  "[alpha_training.pdf, page 1].",
        "citations": [{"filename": "alpha_training.pdf", "page": 1}],
        "insufficient_evidence": False,
    })
    gen, client = _generator(seeded_db, fake_embedder, [payload])
    answer = gen.ask("How often does the Nova plan train each week?")
    assert not answer.insufficient_evidence
    assert "3 sessions" in answer.text
    assert answer.citations and answer.citations[0].filename == "alpha_training.pdf"
    assert answer.citations[0].page == 1
    assert answer.structured


def test_invented_citations_are_dropped(seeded_db, fake_embedder):
    payload = json.dumps({
        "answer": "The answer is 42 [ghost.pdf, page 99].",
        "citations": [
            {"filename": "ghost.pdf", "page": 99},
            {"filename": "alpha_training.pdf", "page": 1},
        ],
        "insufficient_evidence": False,
    })
    gen, _ = _generator(seeded_db, fake_embedder, [payload])
    answer = gen.ask("How often does the Nova plan train each week?")
    # Only the evidence-backed citation survives; ghost.pdf is removed.
    assert [c.filename for c in answer.citations] == ["alpha_training.pdf"]
    assert any("ghost.pdf" in w for w in answer.warnings)


def test_inline_invented_citation_in_text_dropped(seeded_db, fake_embedder):
    payload = json.dumps({
        "answer": "It says so on page 7 [fake_doc.pdf, page 7].",
        "citations": [],
        "insufficient_evidence": False,
    })
    gen, _ = _generator(seeded_db, fake_embedder, [payload])
    answer = gen.ask("How often does the Nova plan train each week?")
    assert answer.citations == []
    assert answer.warnings


def test_abstention_without_llm_call(seeded_db, fake_embedder):
    gen, client = _generator(seeded_db, fake_embedder, [])
    answer = gen.ask("What is the airspeed velocity of an unladen swallow?")
    assert answer.insufficient_evidence
    assert answer.text == config.ABSTENTION_MESSAGE
    assert client.calls == [], "LLM must not be called without evidence"


def test_model_declared_insufficient_evidence(seeded_db, fake_embedder):
    payload = json.dumps({
        "answer": config.ABSTENTION_MESSAGE,
        "citations": [],
        "insufficient_evidence": True,
    })
    gen, _ = _generator(seeded_db, fake_embedder, [payload])
    answer = gen.ask("How often does the Nova plan train each week?")
    assert answer.insufficient_evidence
    assert answer.text == config.ABSTENTION_MESSAGE
    assert answer.citations == []


def test_invalid_json_triggers_retry_then_fallback(seeded_db, fake_embedder):
    gen, client = _generator(
        seeded_db, fake_embedder,
        ["This is not JSON at all.", "Still not JSON, sorry."],
    )
    answer = gen.ask("How often does the Nova plan train each week?")
    assert not answer.structured
    assert answer.citations == [], "fallback must never fabricate citations"
    assert len(client.calls) == 2, "expected exactly one retry"
    assert any("unstructured" in w for w in answer.warnings)


def test_markdown_fenced_json_is_accepted(seeded_db, fake_embedder):
    payload = "```json\n" + json.dumps({
        "answer": "3 sessions weekly [alpha_training.pdf, page 1].",
        "citations": [{"filename": "alpha_training.pdf", "page": 1}],
        "insufficient_evidence": False,
    }) + "\n```"
    gen, _ = _generator(seeded_db, fake_embedder, [payload])
    answer = gen.ask("How often does the Nova plan train each week?")
    assert answer.structured
    assert answer.citations and answer.citations[0].page == 1


def test_history_never_sent_to_model(seeded_db, fake_embedder):
    payload = json.dumps({
        "answer": "3 sessions per week [alpha_training.pdf, page 1].",
        "citations": [{"filename": "alpha_training.pdf", "page": 1}],
        "insufficient_evidence": False,
    })
    gen, client = _generator(seeded_db, fake_embedder, [payload, payload])
    gen.ask("How often does the Nova plan train each week?")
    gen.ask("And in the second question?")
    assert len(client.calls) == 2
    for call in client.calls:
        roles = [m["role"] for m in call]
        assert roles == ["system", "user"], f"unexpected roles: {roles}"
    # The first call must not contain the second question: history is never evidence.
    assert "second question" not in json.dumps(client.calls[0])


def test_evidence_in_prompt_is_from_retrieved_chunks(seeded_db, fake_embedder):
    payload = json.dumps({
        "answer": "ok [alpha_training.pdf, page 1].",
        "citations": [{"filename": "alpha_training.pdf", "page": 1}],
        "insufficient_evidence": False,
    })
    gen, client = _generator(seeded_db, fake_embedder, [payload])
    gen.ask("How often does the Nova plan train each week?")
    user_msg = client.calls[0][1]["content"]
    assert "alpha_training.pdf, page 1" in user_msg
    assert "EXCERPTS" in user_msg


def test_validate_model_output_with_no_evidence():
    answer, cites, insufficient, warnings = validate_model_output(
        json.dumps({
            "answer": "Because [a.pdf, page 1].",
            "citations": [{"filename": "a.pdf", "page": 1}],
            "insufficient_evidence": False,
        }),
        [],  # no evidence -> everything is fabricated
    )
    assert cites == []
    assert warnings
