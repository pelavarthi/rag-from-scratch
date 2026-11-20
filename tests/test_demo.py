"""Tests for the end-to-end demo script."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_demo():
    spec = importlib.util.spec_from_file_location("demo", SCRIPTS / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = load_demo()


def test_demo_passes_on_sample_data():
    result = demo.run_demo("data/sample", "data/sample/eval.json")
    assert len(result.checks) == 6  # six golden questions
    assert result.passed
    for check in result.checks:
        assert check.recall == 1.0
        assert check.citation_precision == 1.0
        assert check.faithfulness == 1.0
        assert check.answer.citations  # every answer cites something
    assert "6/6 checks passed" in result.format()


def test_demo_check_fails_when_retrieval_misses():
    # Gold chunks that exist nowhere in the index can never be
    # retrieved, so the recall check has to fail.
    from rag.eval import ChunkRef, GoldenQuestion

    retriever = demo.build_retriever("data/sample")
    question = GoldenQuestion(
        question="what is the capital of france?",
        answer="Paris.",
        relevant=[ChunkRef(doc_id="no-such-doc.md", chunk_index=99)],
    )
    check = demo.check_question(retriever, question)
    assert check.recall == 0.0
    assert not check.passed


def test_ask_one_question_returns_cited_answer():
    retriever = demo.build_retriever("data/sample")
    answer, results = demo.answer_question(
        retriever, "what is cosine similarity?"
    )
    assert results
    assert answer.citations
    assert all(c.number >= 1 for c in answer.citations)


@pytest.mark.parametrize("path", ["demo.py"])
def test_script_has_no_syntax_errors(path):
    source = (SCRIPTS / path).read_text(encoding="utf-8")
    compile(source, path, "exec")
