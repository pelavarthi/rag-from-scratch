"""Tests for the sentence-aware chunker."""

from rag.chunking import chunk_document, split_into_chunks, split_sentences
from rag.document import Document


def _doc(text):
    return Document(id="d1", title="Test", source="test.txt", text=text)


def test_split_sentences_keeps_punctuation():
    assert split_sentences("Hello world. How are you? Fine!") == [
        "Hello world.",
        "How are you?",
        "Fine!",
    ]


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("") == []
    assert split_into_chunks("   ") == []


def test_short_text_is_one_chunk():
    chunks = split_into_chunks("One short sentence.")
    assert len(chunks) == 1
    assert chunks[0][0] == "One short sentence."
    assert chunks[0][1] == 0
    assert chunks[0][2] == len("One short sentence.")


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} here." for i in range(20))
    chunks = split_into_chunks(text, size=60, overlap=0)
    for chunk_text, _, _ in chunks:
        assert len(chunk_text) <= 60, chunk_text
    # Every sentence appears in at least one chunk.
    joined = " ".join(c[0] for c in chunks)
    for i in range(20):
        assert f"Sentence number {i} here." in joined


def test_overlap_repeats_text():
    text = " ".join(f"Sentence number {i} here." for i in range(10))
    no_overlap = split_into_chunks(text, size=60, overlap=0)
    with_overlap = split_into_chunks(text, size=60, overlap=30)
    assert len(with_overlap) >= len(no_overlap)
    # With overlap, some sentence shows up in two chunks.
    seen: dict[str, int] = {}
    for chunk_text, _, _ in with_overlap:
        for sentence in split_sentences(chunk_text):
            seen[sentence] = seen.get(sentence, 0) + 1
    assert max(seen.values()) > 1


def test_offsets_point_back_into_document():
    text = "First sentence here. Second sentence here. Third one too."
    for chunk_text, start, end in split_into_chunks(text, size=25, overlap=0):
        assert text[start:end].strip() in chunk_text or chunk_text in text


def test_oversized_sentence_becomes_its_own_chunk():
    long_sentence = "word " * 100 + "."
    chunks = split_into_chunks(long_sentence + " Short one.", size=50, overlap=0)
    assert len(chunks) == 2
    assert chunks[0][0] == long_sentence.strip()
    assert chunks[1][0] == "Short one."


def test_invalid_args_rejected():
    import pytest

    with pytest.raises(ValueError):
        split_into_chunks("text", size=0)
    with pytest.raises(ValueError):
        split_into_chunks("text", size=50, overlap=50)


def test_chunk_document_tags_source():
    doc = _doc("Alpha one. Beta two. Gamma three.")
    chunks = chunk_document(doc, size=20, overlap=0)
    assert chunks
    assert all(c.doc_id == "d1" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["title"] == "Test" for c in chunks)
