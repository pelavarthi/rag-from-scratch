"""Tests for chunking v2: recursive, semantic, and dedup."""

import pytest

from rag.chunking import split_sentences
from rag.chunking_v2 import (
    chunk_document_v2,
    deduplicate_chunks,
    normalize_text,
    split_recursive,
    split_semantic,
    text_fingerprint,
)
from rag.document import Chunk, Document
from rag.embeddings import HashingEmbedder
from rag.pipeline import build_chunks


def _doc(text):
    return Document(id="d1", title="Test", source="test.txt", text=text)


def _chunk(text, index=0):
    return Chunk(doc_id="d1", text=text, index=index, start_char=0, end_char=len(text))


# --- recursive splitting ---


def test_recursive_empty_text_gives_no_chunks():
    assert split_recursive("") == []
    assert split_recursive("   ") == []


def test_recursive_short_text_is_one_chunk():
    chunks = split_recursive("One short paragraph.")
    assert len(chunks) == 1
    assert chunks[0][0] == "One short paragraph."
    assert chunks[0][1] == 0
    assert chunks[0][2] == len("One short paragraph.")


def test_recursive_keeps_paragraphs_together():
    first = "First paragraph about cats and dogs."
    second = "Second paragraph about birds and fish."
    chunks = split_recursive(f"{first}\n\n{second}", size=40, overlap=0)
    assert [c[0] for c in chunks] == [first, second]


def test_recursive_respects_size_limit():
    text = "\n\n".join(f"Paragraph number {i} with some words here." for i in range(10))
    chunks = split_recursive(text, size=50, overlap=0)
    for chunk_text, _, _ in chunks:
        assert len(chunk_text) <= 50, chunk_text
    joined = " ".join(c[0] for c in chunks)
    for i in range(10):
        assert f"Paragraph number {i}" in joined


def test_recursive_offsets_point_back_into_document():
    text = "Alpha paragraph here.\n\nBeta paragraph here.\n\nGamma one too."
    for chunk_text, start, end in split_recursive(text, size=25, overlap=0):
        assert text[start:end] == chunk_text


def test_recursive_long_word_is_cut_to_size():
    long_word = "x" * 120
    chunks = split_recursive(long_word + " Short one.", size=50, overlap=0)
    assert chunks[0][0] == "x" * 50
    assert all(len(c[0]) <= 50 for c in chunks)


def test_recursive_overlap_repeats_text():
    text = " ".join(f"Sentence number {i} here." for i in range(10))
    no_overlap = split_recursive(text, size=60, overlap=0)
    with_overlap = split_recursive(text, size=60, overlap=30)
    assert len(with_overlap) >= len(no_overlap)
    # With overlap, some sentence shows up in two chunks.
    seen: dict[str, int] = {}
    for chunk_text, _, _ in with_overlap:
        for sentence in split_sentences(chunk_text):
            seen[sentence] = seen.get(sentence, 0) + 1
    assert max(seen.values()) > 1
    # Without overlap, no sentence repeats.
    seen_no: dict[str, int] = {}
    for chunk_text, _, _ in no_overlap:
        for sentence in split_sentences(chunk_text):
            seen_no[sentence] = seen_no.get(sentence, 0) + 1
    assert max(seen_no.values()) == 1


def test_recursive_invalid_args_rejected():
    with pytest.raises(ValueError):
        split_recursive("text", size=0)
    with pytest.raises(ValueError):
        split_recursive("text", size=50, overlap=50)


# --- semantic splitting ---


def test_semantic_empty_text_gives_no_chunks():
    assert split_semantic("", HashingEmbedder()) == []


def test_semantic_splits_on_topic_change():
    embedder = HashingEmbedder()
    text = (
        "The cat sleeps on the mat. The cat sleeps all afternoon. "
        "The quantum particles behave strangely. "
        "The quantum particles collapse on measurement."
    )
    chunks = split_semantic(text, embedder, threshold=0.5)
    assert len(chunks) == 2
    assert "cat" in chunks[0][0]
    assert "quantum" in chunks[1][0]


def test_semantic_keeps_one_topic_together():
    embedder = HashingEmbedder()
    text = "The cat sleeps on the mat. The cat sleeps all afternoon. The cat sleeps and dreams of fish."
    chunks = split_semantic(text, embedder, threshold=0.5)
    assert len(chunks) == 1
    assert chunks[0][0] == text


def test_semantic_max_size_caps_a_chunk():
    embedder = HashingEmbedder()
    sentence = "The cat sleeps on the mat."
    text = " ".join([sentence] * 6)
    chunks = split_semantic(text, embedder, threshold=0.5, max_size=60)
    assert len(chunks) > 1
    for chunk_text, _, _ in chunks:
        assert len(chunk_text) <= 60


def test_semantic_offsets_point_back_into_document():
    embedder = HashingEmbedder()
    text = "The cat sleeps on the mat. Quantum particles behave strangely."
    for chunk_text, start, end in split_semantic(text, embedder, threshold=0.5):
        assert text[start:end] == chunk_text


def test_semantic_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        split_semantic("text.", HashingEmbedder(), threshold=1.5)


# --- dedup ---


def test_dedup_drops_exact_copies():
    chunks = [_chunk("Hello world."), _chunk("Hello world."), _chunk("Other text.")]
    kept = deduplicate_chunks(chunks)
    assert [c.text for c in kept] == ["Hello world.", "Other text."]


def test_dedup_ignores_case_and_spacing():
    chunks = [_chunk("Hello   world."), _chunk("  hello world. ")]
    assert len(deduplicate_chunks(chunks)) == 1


def test_dedup_keeps_first_and_preserves_order():
    chunks = [_chunk("First."), _chunk("Second."), _chunk("First.")]
    kept = deduplicate_chunks(chunks)
    assert [c.text for c in kept] == ["First.", "Second."]


def test_dedup_drops_near_copies_with_embedder():
    chunks = [_chunk("Cats sleep all day."), _chunk("Cats sleep all day!")]
    kept = deduplicate_chunks(chunks, embedder=HashingEmbedder())
    assert len(kept) == 1


def test_dedup_keeps_distinct_chunks():
    chunks = [_chunk("Cats sleep all day."), _chunk("Quantum particles behave strangely.")]
    kept = deduplicate_chunks(chunks, embedder=HashingEmbedder())
    assert len(kept) == 2


def test_fingerprint_is_stable_and_case_insensitive():
    assert text_fingerprint("Hello World.") == text_fingerprint("  hello   world. ")
    assert normalize_text("Hello   World.") == "hello world."


# --- document wrapper + pipeline ---


def test_chunk_document_v2_tags_strategy():
    doc = _doc("First paragraph here.\n\nSecond paragraph here.")
    chunks = chunk_document_v2(doc, strategy="recursive", size=30, overlap=0)
    assert chunks
    assert all(c.metadata["strategy"] == "recursive" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_v2_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        chunk_document_v2(_doc("text."), strategy="magic")


def test_chunk_document_v2_semantic_needs_embedder():
    with pytest.raises(ValueError):
        chunk_document_v2(_doc("text."), strategy="semantic")


def test_pipeline_recursive_strategy_on_sample_docs():
    chunks = build_chunks("data/sample", strategy="recursive")
    assert chunks
    assert all(c.metadata["strategy"] == "recursive" for c in chunks)


def test_pipeline_semantic_strategy_on_sample_docs():
    chunks = build_chunks("data/sample", strategy="semantic")
    assert chunks
    assert all(c.metadata["strategy"] == "semantic" for c in chunks)


def test_pipeline_dedup_removes_repeated_chunks():
    chunks = build_chunks("data/sample", strategy="recursive", dedup=True)
    fingerprints = [text_fingerprint(c.text) for c in chunks]
    assert len(fingerprints) == len(set(fingerprints))
