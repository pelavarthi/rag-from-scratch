"""Tests for the evaluation harness: metrics, golden sets, reports."""

from pathlib import Path

import pytest

from rag.answers import Answer
from rag.document import Chunk
from rag.eval import (
    AnswerRow,
    ChunkRef,
    EvalSet,
    GoldenQuestion,
    RetrievalReport,
    RetrievalRow,
    answer_overlap,
    citation_precision,
    content_words,
    evaluate_answers,
    evaluate_retrieval,
    faithfulness,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    split_claims,
)
from rag.vector_store import SearchResult

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def _chunk(doc_id, index, text):
    return Chunk(
        doc_id=doc_id,
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
        metadata={"title": doc_id},
    )


def _result(doc_id, index, text, score=0.9):
    return SearchResult(chunk=_chunk(doc_id, index, text), score=score)


def _refs(*pairs):
    return [ChunkRef(doc_id=doc_id, chunk_index=i) for doc_id, i in pairs]


def test_recall_at_k_counts_found_relevant_chunks():
    results = [_result("a", 0, "x"), _result("b", 1, "y"), _result("c", 2, "z")]
    relevant = _refs(("a", 0), ("c", 2), ("d", 0))
    assert recall_at_k(results, relevant, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(results, relevant, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(results, relevant, k=5) == pytest.approx(2 / 3)


def test_recall_at_k_with_no_relevant_chunks_is_one():
    results = [_result("a", 0, "x")]
    assert recall_at_k(results, [], k=3) == 1.0


def test_recall_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k([_result("a", 0, "x")], _refs(("a", 0)), k=0)


def test_precision_at_k_counts_relevant_share_of_top_k():
    results = [_result("a", 0, "x"), _result("b", 1, "y"), _result("c", 2, "z")]
    relevant = _refs(("a", 0), ("c", 2))
    assert precision_at_k(results, relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(results, relevant, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_with_no_results_is_zero():
    assert precision_at_k([], _refs(("a", 0)), k=3) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit():
    results = [_result("a", 0, "x"), _result("b", 1, "y"), _result("c", 2, "z")]
    assert reciprocal_rank(results, _refs(("a", 0))) == 1.0
    assert reciprocal_rank(results, _refs(("b", 1))) == pytest.approx(0.5)
    assert reciprocal_rank(results, _refs(("d", 0))) == 0.0


def test_content_words_drops_stopwords_and_lowercases():
    assert content_words("The Cat sat on the mat") == {"cat", "sat", "mat"}


def test_split_claims_splits_on_sentence_ends():
    assert split_claims("First claim. Second claim! Third?") == [
        "First claim.",
        "Second claim!",
        "Third?",
    ]


def test_faithfulness_is_one_when_every_word_is_in_context():
    context = [_chunk("a", 0, "cats sleep all day in the sun")]
    assert faithfulness("Cats sleep all day. [1]", context) == 1.0


def test_faithfulness_drops_when_the_answer_invents_words():
    context = [_chunk("a", 0, "cats sleep all day")]
    # "quantum" and "entanglement" appear nowhere in the context.
    score = faithfulness("Cats sleep all day. Quantum entanglement rules.", context)
    assert 0.0 < score < 1.0


def test_faithfulness_ignores_abstentions_and_empty_answers():
    context = [_chunk("a", 0, "cats sleep all day")]
    assert faithfulness("I do not know.", context) == 1.0
    assert faithfulness("", context) == 1.0


def test_citation_precision_counts_valid_citations():
    from rag.answers import build_prompt

    prompt = build_prompt("q?", [_result("a", 0, "x"), _result("b", 1, "y")])
    all_valid = Answer(text="one [1] two [2]", citations=[])
    assert citation_precision(all_valid, prompt.citations) == 1.0
    half_valid = Answer(text="one [1] two [99]", citations=[])
    assert citation_precision(half_valid, prompt.citations) == 0.5


def test_citation_precision_is_none_when_nothing_is_cited():
    from rag.answers import build_prompt

    prompt = build_prompt("q?", [_result("a", 0, "x")])
    assert citation_precision(Answer(text="no citations", citations=[]), prompt.citations) is None


def test_answer_overlap_measures_coverage_of_reference():
    assert answer_overlap("cats sleep all day", "cats sleep all day") == 1.0
    assert answer_overlap("cats sleep", "cats sleep all day") == pytest.approx(2 / 3)
    assert answer_overlap("dogs bark", "cats sleep all day") == 0.0
    assert answer_overlap("anything", "") == 1.0


def test_eval_set_round_trips_through_json(tmp_path):
    path = tmp_path / "eval.json"
    original = EvalSet(
        questions=[
            GoldenQuestion(
                question="what is rag?",
                answer="retrieval-augmented generation",
                relevant=[ChunkRef("a.md", 0), ChunkRef("b.md", 2)],
            )
        ]
    )
    original.save(path)
    loaded = EvalSet.load(path)
    assert len(loaded) == 1
    assert loaded.questions[0].question == "what is rag?"
    assert loaded.questions[0].relevant == [ChunkRef("a.md", 0), ChunkRef("b.md", 2)]


def _tiny_retriever(tmp_path):
    """A hybrid retriever over two tiny docs, for end-to-end eval tests."""
    from rag.embeddings import HashingEmbedder
    from rag.hybrid import HybridRetriever
    from rag.pipeline import build_chunks

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text(
        "# RAG\n\nRetrieval-augmented generation answers questions from "
        "documents instead of guessing.\n\nIt retrieves passages first, "
        "then generates an answer from them.\n"
    )
    (docs / "bread.txt").write_text(
        "Baking bread needs flour, water, salt, and yeast. "
        "Knead the dough and let it rise before baking.\n"
    )
    chunks = build_chunks(docs)
    return HybridRetriever.from_chunks(chunks, HashingEmbedder(dim=256)), chunks


def test_evaluate_retrieval_finds_the_right_doc(tmp_path):
    retriever, chunks = _tiny_retriever(tmp_path)
    rag_chunk = next(c for c in chunks if c.doc_id == "rag.md")
    eval_set = EvalSet(
        questions=[
            GoldenQuestion(
                question="what is retrieval-augmented generation?",
                answer="answers questions from documents",
                relevant=[ChunkRef("rag.md", rag_chunk.index)],
            )
        ]
    )
    report = evaluate_retrieval(retriever, eval_set, ks=(1, 3, 5))
    assert report.mean_recall_at_k[5] == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.rows[0].recall_at_k[1] <= report.rows[0].recall_at_k[5]


def test_evaluate_retrieval_rejects_empty_ks(tmp_path):
    retriever, _ = _tiny_retriever(tmp_path)
    with pytest.raises(ValueError):
        evaluate_retrieval(retriever, EvalSet(), ks=())


def test_retrieval_report_means_average_rows():
    rows = [
        RetrievalRow("q1", {1: 1.0, 3: 1.0}, {1: 1.0, 3: 0.5}, 1.0),
        RetrievalRow("q2", {1: 0.0, 3: 0.5}, {1: 0.0, 3: 0.25}, 0.5),
    ]
    report = RetrievalReport(rows=rows, ks=(1, 3))
    assert report.mean_recall_at_k == {1: 0.5, 3: 0.75}
    assert report.mean_precision_at_k == {1: 0.5, 3: 0.375}
    assert report.mean_reciprocal_rank == pytest.approx(0.75)
    assert "mean recall@3: 0.750" in report.format()


def test_answer_report_mean_citation_precision_skips_uncited():
    rows = [
        AnswerRow("q1", faithfulness=1.0, citation_precision=1.0, answer_overlap=0.5),
        AnswerRow("q2", faithfulness=0.5, citation_precision=None, answer_overlap=0.0),
    ]
    from rag.eval import AnswerReport

    report = AnswerReport(rows=rows)
    assert report.mean_faithfulness == pytest.approx(0.75)
    assert report.mean_citation_precision == 1.0
    assert report.mean_answer_overlap == pytest.approx(0.25)
    assert "mean citation precision: 1.000" in report.format()


def test_evaluate_answers_end_to_end_with_stub_model(tmp_path):
    from rag.answers import AnswerGenerator

    retriever, _ = _tiny_retriever(tmp_path)
    eval_set = EvalSet(
        questions=[
            GoldenQuestion(
                question="what is retrieval-augmented generation?",
                answer="answers questions from documents instead of guessing",
                relevant=[ChunkRef("rag.md", 0)],
            )
        ]
    )

    def stub(prompt):
        passage = prompt.split("\n[1] ", 1)[1].split("\n\nQuestion:")[0]
        first_sentence = passage.strip().split(". ")[0]
        return f"{first_sentence}. [1]"

    report = evaluate_answers(
        AnswerGenerator(generate=stub).answer, retriever, eval_set, k=3
    )
    assert len(report.rows) == 1
    assert report.rows[0].faithfulness == 1.0
    assert report.rows[0].citation_precision == 1.0
    assert report.rows[0].answer_overlap > 0.0


def test_sample_eval_set_scores_are_sane():
    """The shipped golden set over the sample docs, end to end."""
    from rag.embeddings import HashingEmbedder
    from rag.hybrid import HybridRetriever
    from rag.pipeline import build_chunks

    eval_set = EvalSet.load(SAMPLE_DIR / "eval.json")
    assert len(eval_set) == 6
    chunks = build_chunks(SAMPLE_DIR)
    retriever = HybridRetriever.from_chunks(chunks, HashingEmbedder(dim=256))
    report = evaluate_retrieval(retriever, eval_set, ks=(1, 3, 5))
    # The hashing embedder is a baseline, so the bar is modest: at
    # least half the expected chunks should surface in the top 5, and
    # recall must not shrink as k grows.
    assert report.mean_recall_at_k[5] >= 0.5
    assert report.mean_recall_at_k[1] <= report.mean_recall_at_k[3]
    assert report.mean_recall_at_k[3] <= report.mean_recall_at_k[5]
    assert report.mean_reciprocal_rank > 0.0
