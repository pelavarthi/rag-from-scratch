"""Tests for retrieval: metadata filters, MMR diversity, top-k."""

import pytest

from rag.document import Chunk
from rag.embeddings import HashingEmbedder
from rag.pipeline import build_chunks
from rag.retrieval import Retriever, matches_metadata, mmr_rerank
from rag.vector_store import SearchResult, VectorStore


def _chunk(text, doc_id="d1", index=0, metadata=None):
    return Chunk(
        doc_id=doc_id,
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
        metadata=metadata or {},
    )


def _retriever(texts, dim=256, **kwargs):
    enc = HashingEmbedder(dim=dim)
    chunks = [_chunk(t, index=i) for i, t in enumerate(texts)]
    store = VectorStore.from_chunks(chunks, enc)
    return Retriever(store, enc, **kwargs), store, enc


def test_matches_metadata_doc_id_and_fields():
    chunk = _chunk("cats sleep", doc_id="d1", metadata={"title": "Cats", "source": "cats.md"})
    assert matches_metadata(chunk, {"doc_id": "d1"})
    assert matches_metadata(chunk, {"title": "Cats"})
    assert matches_metadata(chunk, {"doc_id": "d1", "title": "Cats", "source": "cats.md"})
    assert not matches_metadata(chunk, {"doc_id": "d2"})
    assert not matches_metadata(chunk, {"title": "Dogs"})
    assert not matches_metadata(chunk, {"author": "someone"})


def test_matches_metadata_list_value_matches_any():
    chunk = _chunk("cats sleep", doc_id="d1", metadata={"title": "Cats"})
    assert matches_metadata(chunk, {"doc_id": ["d1", "d2"]})
    assert matches_metadata(chunk, {"title": ["Dogs", "Cats"]})
    assert not matches_metadata(chunk, {"doc_id": ["d2", "d3"]})
    assert not matches_metadata(chunk, {"title": ("Dogs", "Birds")})


def test_retrieve_filters_by_doc_id():
    enc = HashingEmbedder(dim=256)
    chunks = [
        _chunk("cats sleep all day in the sun", doc_id="cats", index=0),
        _chunk("cats nap in the warm afternoon sun", doc_id="cats", index=1),
        _chunk("cats purr softly on the sofa", doc_id="dogs", index=0),
    ]
    store = VectorStore.from_chunks(chunks, enc)
    retriever = Retriever(store, enc)

    results = retriever.retrieve("cats sleep", k=3, where={"doc_id": "dogs"})
    assert len(results) == 1
    assert results[0].chunk.doc_id == "dogs"

    results = retriever.retrieve("cats sleep", k=3, where={"doc_id": "cats"})
    assert len(results) == 2
    assert all(r.chunk.doc_id == "cats" for r in results)


def test_retrieve_filters_by_metadata():
    enc = HashingEmbedder(dim=256)
    chunks = [
        _chunk("cats sleep all day", doc_id="d1", metadata={"title": "Cats"}),
        _chunk("cats nap in the sun", doc_id="d2", metadata={"title": "Cats"}),
        _chunk("cats purr on the sofa", doc_id="d3", metadata={"title": "Dogs"}),
    ]
    store = VectorStore.from_chunks(chunks, enc)
    retriever = Retriever(store, enc)

    results = retriever.retrieve("cats sleep", k=3, where={"title": "Dogs"})
    assert [r.chunk.doc_id for r in results] == ["d3"]


def test_retrieve_filter_no_match_returns_empty():
    retriever, _, _ = _retriever(["cats sleep all day", "dogs run in the park"])
    assert retriever.retrieve("cats", where={"doc_id": "nope"}) == []


def test_retrieve_accepts_callable_where():
    retriever, _, _ = _retriever(["cats sleep all day", "dogs run in the park"])
    results = retriever.retrieve(
        "cats sleep", k=2, where=lambda c: "dogs" in c.text
    )
    assert [r.chunk.text for r in results] == ["dogs run in the park"]


def test_retrieve_rejects_bad_where_type():
    retriever, _, _ = _retriever(["cats sleep"])
    with pytest.raises(TypeError):
        retriever.retrieve("cats", where=42)


def test_retrieve_returns_at_most_k():
    retriever, _, _ = _retriever(["cats sleep", "dogs run", "birds fly"])
    assert len(retriever.retrieve("cats", k=2)) == 2


def test_retrieve_k_larger_than_store_returns_all():
    retriever, _, _ = _retriever(["cats sleep", "dogs run"])
    assert len(retriever.retrieve("cats", k=10)) == 2


def test_retrieve_empty_store_returns_nothing():
    store = VectorStore(dim=256)
    retriever = Retriever(store, HashingEmbedder(dim=256))
    assert retriever.retrieve("cats") == []


def test_retrieve_rejects_bad_k_and_lambda():
    retriever, _, _ = _retriever(["cats sleep"])
    with pytest.raises(ValueError):
        retriever.retrieve("cats", k=0)
    with pytest.raises(ValueError):
        retriever.retrieve("cats", lambda_mult=-0.1)
    with pytest.raises(ValueError):
        retriever.retrieve("cats", lambda_mult=1.5)


def test_retriever_rejects_mismatched_embedder():
    store = VectorStore(dim=256)
    with pytest.raises(ValueError):
        Retriever(store, HashingEmbedder(dim=64))


def test_retriever_rejects_bad_lambda_and_fetch_multiplier():
    retriever, _, _ = _retriever(["cats sleep"])
    with pytest.raises(ValueError):
        Retriever(retriever.store, retriever.embedder, lambda_mult=2.0)
    with pytest.raises(ValueError):
        Retriever(retriever.store, retriever.embedder, fetch_multiplier=0)


def test_mmr_lambda_one_matches_top_k():
    retriever, store, enc = _retriever(
        [
            "cats sleep all day in the sun",
            "cats sleep all day in the warm sun",
            "dogs run in the park every morning",
        ]
    )
    top_k = store.search(enc.embed("cats sleep all day"), k=2)
    reranked = retriever.retrieve("cats sleep all day", k=2, lambda_mult=1.0)
    assert [r.chunk.text for r in reranked] == [r.chunk.text for r in top_k]


def test_mmr_diversity_prefers_different_second_hit():
    retriever, _, _ = _retriever(
        [
            "cats sleep all day in the sun",
            "cats sleep all day in the warm sun",
            "dogs run in the park every morning",
        ]
    )
    pure_relevance = retriever.retrieve("cats sleep all day", k=2, lambda_mult=1.0)
    assert all("cats" in r.chunk.text for r in pure_relevance)

    diverse = retriever.retrieve("cats sleep all day", k=2, lambda_mult=0.0)
    assert diverse[0].chunk.text == pure_relevance[0].chunk.text
    assert diverse[1].chunk.text.startswith("dogs run")


def test_mmr_rerank_validates_inputs():
    retriever, store, enc = _retriever(["cats sleep"])
    candidates = store.search_with_vectors(enc.embed("cats"))
    with pytest.raises(ValueError):
        mmr_rerank(enc.embed("cats"), candidates, k=0)
    with pytest.raises(ValueError):
        mmr_rerank(enc.embed("cats"), candidates, k=1, lambda_mult=1.5)
    assert mmr_rerank(enc.embed("cats"), [], k=3) == []


def test_mmr_rerank_returns_search_results():
    retriever, store, enc = _retriever(["cats sleep all day", "dogs run"])
    candidates = store.search_with_vectors(enc.embed("cats"), k=2)
    ranked = mmr_rerank(enc.embed("cats"), candidates, k=2)
    assert all(isinstance(r, SearchResult) for r in ranked)
    assert ranked[0].chunk.text.startswith("cats sleep")


def test_search_with_vectors_matches_search_order():
    retriever, store, enc = _retriever(
        ["cats sleep all day", "quantum physics lectures", "dogs run"]
    )
    query = enc.embed("cats nap all day")
    plain = store.search(query, k=3)
    detailed = store.search_with_vectors(query, k=3)
    assert [r.chunk.text for r, _ in detailed] == [r.chunk.text for r in plain]
    assert [v for _, v in detailed] == [
        enc.embed("cats sleep all day"),
        enc.embed("quantum physics lectures"),
        enc.embed("dogs run"),
    ]


def test_store_search_where_filters_before_ranking():
    retriever, store, enc = _retriever(["cats sleep", "dogs run"])
    results = store.search(enc.embed("cats"), where=lambda c: c.index == 1)
    assert [r.chunk.text for r in results] == ["dogs run"]


def test_fetch_k_smaller_than_k_is_raised_to_k():
    retriever, _, _ = _retriever(
        ["cats sleep", "dogs run", "birds fly", "fish swim", "horses gallop"]
    )
    assert len(retriever.retrieve("cats", k=3, fetch_k=1)) == 3


def test_retrieve_over_sample_docs_with_filter(tmp_path):
    chunks = build_chunks("data/sample")
    enc = HashingEmbedder(dim=256)
    store = VectorStore.from_chunks(chunks, enc)
    retriever = Retriever(store, enc)

    results = retriever.retrieve("vector search", k=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)

    filtered = retriever.retrieve(
        "vector search", k=5, where={"doc_id": "vector-search.txt"}
    )
    assert filtered
    assert all(r.chunk.doc_id == "vector-search.txt" for r in filtered)
