import asyncio
import uuid
from types import SimpleNamespace as NS

import pytest

from app.services.answers import (
    AnswerUnavailable,
    ClaudeAnswerer,
    Source,
    assemble_answer,
    build_documents,
    source_excerpt,
)

SOURCES = [
    Source(uuid.uuid4(), "Q3 planning", "decision", "2026-09-01", "Mobile launch moves to January."),
    Source(uuid.uuid4(), "Payments sync", "meeting", "2026-09-03", "Payments team is short two engineers."),
]


def text(t, *cites):
    return NS(type="text", text=t, citations=[NS(document_index=i, cited_text=c) for i, c in cites] or None)


def test_build_documents_enables_citations_on_every_source():
    docs = build_documents(SOURCES)
    assert [d["title"] for d in docs] == ["Q3 planning", "Payments sync"]
    assert all(d["citations"] == {"enabled": True} for d in docs)
    assert docs[0]["source"] == {"type": "text", "media_type": "text/plain", "data": SOURCES[0].text}


def test_assemble_answer_places_markers_and_dedupes_citations():
    content = [
        NS(type="thinking", thinking=""),
        text("The launch moved to January", (0, "Mobile launch moves to January.")),
        text(" because payments is understaffed", (1, "Payments team is short two engineers."), (1, "short two")),
        text("."),
        text(" Again, January", (0, "Mobile launch moves to January.")),
    ]
    answer, citations = assemble_answer(content)
    assert answer == "The launch moved to January[1] because payments is understaffed[2]. Again, January[1]"
    assert [(c.source, c.cited_text) for c in citations] == [
        (1, "Mobile launch moves to January."),
        (2, "Payments team is short two engineers."),
        (2, "short two"),
    ]


def test_source_excerpt_keeps_short_bodies_and_centres_long_ones():
    assert source_excerpt("short", "short", 100) == "short"
    body = "a" * 5000 + "NEEDLE in the haystack" + "b" * 5000
    excerpt = source_excerpt(body, "NEEDLE in the haystack", 1000)
    assert len(excerpt) == 1000 and "NEEDLE in the haystack" in excerpt


def make_answerer(response=None, error=None):
    answerer = ClaudeAnswerer("claude-opus-5-5", "low", 8000, 30)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return response

    answerer._client = NS(beta=NS(messages=NS(create=create)))
    return answerer, calls


def response(content, stop_reason="end_turn"):
    return NS(
        content=content,
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=NS(input_tokens=100, output_tokens=20),
        _request_id="req_test",
    )


def test_claude_answerer_request_shape_and_parsing():
    answerer, calls = make_answerer(response([text("January", (0, "Mobile launch moves to January."))]))
    result = asyncio.run(answerer.answer("When is the launch?", SOURCES))

    req = calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    content = req["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert content[-1]["text"] == "When is the launch?"
    assert result.text == "January[1]" and result.grounded and result.model == "claude-opus-5-5"


def test_claude_answerer_uncited_answer_is_not_grounded():
    answerer, _ = make_answerer(response([text("The memories don't say.")]))
    result = asyncio.run(answerer.answer("Who won the match?", SOURCES))
    assert not result.grounded and result.citations == []


def test_claude_answerer_handles_refusal():
    answerer, _ = make_answerer(response([], stop_reason="refusal"))
    result = asyncio.run(answerer.answer("q", SOURCES))
    assert not result.grounded and result.text


def test_missing_credentials_become_answer_unavailable():
    err = TypeError("Could not resolve authentication method. Expected one of api_key, auth_token...")
    answerer, _ = make_answerer(error=err)
    with pytest.raises(AnswerUnavailable):
        asyncio.run(answerer.answer("q", SOURCES))


def test_unrelated_type_errors_are_not_swallowed():
    answerer, _ = make_answerer(error=TypeError("unexpected keyword argument"))
    with pytest.raises(TypeError):
        asyncio.run(answerer.answer("q", SOURCES))
