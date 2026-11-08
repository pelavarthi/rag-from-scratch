"""Tests for the embedding interface and the hashing baseline encoder."""

import math

import pytest

from rag.document import Chunk
from rag.embeddings import (
    HashingEmbedder,
    cosine_similarity,
    embed_chunks,
    normalize,
)


def _chunk(text, doc_id="d1"):
    return Chunk(doc_id=doc_id, text=text, index=0, start_char=0, end_char=len(text))


def test_embed_is_deterministic():
    enc = HashingEmbedder()
    assert enc.embed("vector search finds similar text") == enc.embed(
        "vector search finds similar text"
    )


def test_embed_has_declared_dim_and_unit_length():
    enc = HashingEmbedder(dim=64)
    vec = enc.embed("some words here")
    assert len(vec) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0)


def test_empty_text_is_zero_vector():
    enc = HashingEmbedder(dim=32)
    vec = enc.embed("")
    assert vec == [0.0] * 32
    assert enc.embed("   ") == [0.0] * 32


def test_case_does_not_change_embedding():
    enc = HashingEmbedder()
    assert enc.embed("Cats and Dogs") == enc.embed("cats and dogs")


def test_different_texts_give_different_vectors():
    enc = HashingEmbedder()
    assert enc.embed("cats sleep all day") != enc.embed("quantum physics lectures")


def test_similar_texts_score_higher_than_unrelated():
    enc = HashingEmbedder()
    query = enc.embed("cats sleep all day")
    same_topic = enc.embed("cats nap in the sun all day")
    other_topic = enc.embed("quantum physics lectures on monday")
    assert cosine_similarity(query, same_topic) > cosine_similarity(query, other_topic)


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])


def test_normalize_zero_vector_stays_zero():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]
    assert normalize([3.0, 4.0]) == [0.6, 0.8]


def test_embed_many_matches_embed_one_by_one():
    enc = HashingEmbedder()
    texts = ["first text", "second text", ""]
    assert enc.embed_many(texts) == [enc.embed(t) for t in texts]


def test_embed_chunks_pairs_chunks_with_vectors():
    enc = HashingEmbedder(dim=16)
    chunks = [_chunk("cats sleep"), _chunk("dogs run")]
    embedded = embed_chunks(chunks, enc)
    assert len(embedded) == 2
    assert embedded[0].chunk is chunks[0]
    assert embedded[0].vector == enc.embed("cats sleep")
    assert len(embedded[1].vector) == 16


def test_embed_chunks_works_with_any_embedder():
    """The interface is pluggable: a stub encoder slots straight in."""

    class CountingEmbedder:
        dim = 3

        def embed(self, text):
            return [float(len(text)), 0.0, 0.0]

        def embed_many(self, texts):
            return [self.embed(t) for t in texts]

    stub = CountingEmbedder()
    assert isinstance(stub, object)
    embedded = embed_chunks([_chunk("hello"), _chunk("hi")], stub)
    assert embedded[0].vector == [5.0, 0.0, 0.0]
    assert embedded[1].vector == [2.0, 0.0, 0.0]


def test_invalid_dim_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
