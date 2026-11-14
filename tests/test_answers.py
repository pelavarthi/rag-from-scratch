"""Tests for answer generation: packing, prompts, citations."""

import pytest

from rag.answers import (
    AnswerGenerator,
    build_prompt,
    format_sources,
    pack_context,
    parse_citations,
    verify_citations,
)
from rag.document import Chunk
from rag.vector_store import SearchResult


def _result(text, doc_id="d1", index=0, metadata=None, score=0.9):
    chunk = Chunk(
        doc_id=doc_id,
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
        metadata=metadata or {"title": "Doc One"},
    )
    return SearchResult(chunk=chunk, score=score)


def test_pack_context_fits_best_first_and_reports_dropped():
    results = [_result("a" * 10), _result("b" * 10), _result("c" * 10)]
    packed, dropped = pack_context(results, max_chars=25)
    assert [r.chunk.text for r in packed] == ["a" * 10, "b" * 10]
    assert dropped == 1


def test_pack_context_truncates_first_chunk_when_it_alone_overflows():
    results = [_result("x" * 100), _result("y" * 5)]
    packed, dropped = pack_context(results, max_chars=20)
    assert len(packed) == 1
    assert packed[0].chunk.text == "x" * 17 + "..."
    assert dropped == 1


def test_pack_context_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        pack_context([_result("hello")], max_chars=0)


def test_build_prompt_numbers_passages_and_asks_question():
    results = [_result("cats sleep all day"), _result("dogs bark at night")]
    prompt = build_prompt("why do cats sleep?", results)
    assert "[1] cats sleep all day" in prompt.text
    assert "[2] dogs bark at night" in prompt.text
    assert "Question: why do cats sleep?" in prompt.text
    assert "cite" in prompt.text.lower()
    assert [c.number for c in prompt.citations] == [1, 2]
    assert prompt.citations[0].doc_id == "d1"


def test_build_prompt_with_no_results_says_so():
    prompt = build_prompt("what is this?", [])
    assert "(no passages were retrieved)" in prompt.text
    assert prompt.citations == []


def test_build_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        build_prompt("   ", [_result("something")])


def test_parse_citations_in_order_without_repeats():
    assert parse_citations("see [2] and [1], then [2] again") == [2, 1]
    assert parse_citations("no citations here") == []


def test_verify_citations_keeps_real_and_flags_invented():
    prompt = build_prompt("q?", [_result("aaa"), _result("bbb")])
    valid, invalid = verify_citations(
        "first [1], then [99], also [0] and [2]", prompt.citations
    )
    assert [c.number for c in valid] == [1, 2]
    assert invalid == [99, 0]


def test_format_sources_lists_each_cited_passage():
    prompt = build_prompt("q?", [_result("aaa", index=3)])
    assert format_sources(prompt.citations) == "Sources:\n[1] Doc One (chunk 3)"
    assert format_sources([]) == "Sources: none"


def test_answer_generator_verifies_citations_from_stub_model():
    results = [_result("cats sleep all day"), _result("dogs bark at night")]

    def stub(prompt):
        assert "[1] cats sleep all day" in prompt
        return "Cats sleep all day [1], and also [99] says otherwise."

    answer = AnswerGenerator(generate=stub).answer("why do cats sleep?", results)
    assert "Cats sleep all day" in answer.text
    assert [c.number for c in answer.citations] == [1]


def test_answer_generator_with_no_results_still_calls_model():
    seen = {}

    def stub(prompt):
        seen["prompt"] = prompt
        return "I do not know."

    answer = AnswerGenerator(generate=stub).answer("what?", [])
    assert answer.text == "I do not know."
    assert answer.citations == []
    assert "(no passages were retrieved)" in seen["prompt"]


def test_answer_end_to_end_with_real_retrieval(tmp_path):
    """Retrieval + answer generation over the sample docs."""
    from rag.embeddings import HashingEmbedder
    from rag.pipeline import build_chunks
    from rag.retrieval import Retriever
    from rag.vector_store import VectorStore

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text(
        "# RAG\n\nRetrieval-augmented generation answers questions from "
        "documents instead of guessing.\n\nIt retrieves passages first, "
        "then generates an answer from them.\n"
    )
    (docs / "other.txt").write_text(
        "Baking bread needs flour, water, salt, and yeast. "
        "Knead the dough and let it rise before baking.\n"
    )
    chunks = build_chunks(docs)
    enc = HashingEmbedder(dim=256)
    store = VectorStore.from_chunks(chunks, enc)
    retriever = Retriever(store, enc)
    results = retriever.retrieve("what is retrieval-augmented generation?", k=2)

    def stub(prompt):
        passage = prompt.split("\n[1] ", 1)[1].split("\n[2] ")[0]
        passage = passage.split("\n\nQuestion:")[0]
        first_sentence = passage.strip().split(". ")[0]
        return f"{first_sentence}. [1]"

    answer = AnswerGenerator(generate=stub).answer(
        "what is retrieval-augmented generation?", results
    )
    assert len(answer.citations) == 1
    assert answer.citations[0].doc_id == "rag.md"
    assert "retrieval-augmented generation" in answer.text.lower()
