"""Tests for hybrid search: BM25, reciprocal rank fusion, HybridRetriever."""

import pytest

from rag.document import Chunk
from rag.embeddings import HashingEmbedder
from rag.hybrid import (
    BM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from rag.retrieval import mmr_rerank
from rag.vector_store import SearchResult, VectorStore


def _chunk(text, doc_id="d", index=0, metadata=None):
    return Chunk(
        doc_id=doc_id,
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
        metadata=metadata or {},
    )


def _hybrid(texts, dim=256, **kwargs):
    enc = HashingEmbedder(dim=dim)
    chunks = [_chunk(t, index=i) for i, t in enumerate(texts)]
    return HybridRetriever.from_chunks(chunks, enc, **kwargs), enc


def test_tokenize_lowercases_and_splits_on_words():
    assert tokenize("Cats, SLEEP! all-day") == ["cats", "sleep", "all", "day"]


def test_bm25_ranks_exact_match_first():
    index = BM25Index.from_chunks(
        [
            _chunk("dogs run in the park every morning", index=0),
            _chunk("cats sleep all day in the sun", index=1),
            _chunk("cats purr softly on the sofa", index=2),
        ]
    )
    results = index.search("cats sleep", k=3)
    assert [r.chunk.index for r in results] == [1, 2]
    assert results[0].score > results[1].score


def test_bm25_no_shared_word_scores_zero_and_is_left_out():
    index = BM25Index.from_chunks(
        [
            _chunk("cats sleep all day", index=0),
            _chunk("dogs run in the park", index=1),
        ]
    )
    assert [r.chunk.index for r in index.search("cats", k=2)] == [0]
    assert index.search("birds") == []
    assert index.search("") == []


def test_bm25_rare_word_beats_common_word():
    # "search" is in every chunk, "quark" in one: the rare word matters more.
    index = BM25Index.from_chunks(
        [
            _chunk("search engines rank pages by relevance and links", index=0),
            _chunk("quark flavors up down strange charm bottom top", index=1),
            _chunk("search search search search search search search search", index=2),
            _chunk("the quark model explains particle physics and search results", index=3),
        ]
    )
    results = index.search("quark search", k=4)
    assert [r.chunk.index for r in results] == [3, 2, 1, 0]
    # The chunk with only the rare word outranks the one with only the common word.
    assert results[2].chunk.index == 1
    assert results[3].chunk.index == 0


def test_bm25_short_chunk_beats_long_chunk_with_same_term():
    index = BM25Index.from_chunks(
        [
            _chunk(
                "quark plus many other words filling up this long chunk "
                "about nothing in particular at all really",
                index=0,
            ),
            _chunk("quark", index=1),
        ]
    )
    results = index.search("quark", k=2)
    assert [r.chunk.index for r in results] == [1, 0]


def test_bm25_repeated_term_scores_higher():
    index = BM25Index.from_chunks(
        [
            _chunk("cats sleep", index=0),
            _chunk("cats cats cats sleep", index=1),
        ]
    )
    results = index.search("cats", k=2)
    assert [r.chunk.index for r in results] == [1, 0]


def test_bm25_search_respects_where_and_k():
    index = BM25Index.from_chunks(
        [
            _chunk("cats sleep all day", index=0),
            _chunk("cats nap in the sun", index=1),
            _chunk("cats purr on the sofa", index=2),
        ]
    )
    assert [r.chunk.index for r in index.search("cats", k=1)] == [0]
    filtered = index.search("cats", where=lambda c: c.index >= 1)
    assert [r.chunk.index for r in filtered] == [1, 2]


def test_bm25_validates_inputs():
    with pytest.raises(ValueError):
        BM25Index(k1=-1)
    with pytest.raises(ValueError):
        BM25Index(b=1.5)
    index = BM25Index.from_chunks([_chunk("cats sleep")])
    with pytest.raises(ValueError):
        index.search("cats", k=0)
    assert BM25Index.from_chunks([]).search("cats") == []


def test_rrf_chunk_second_in_both_beats_first_in_one():
    # Classic RRF property: ranked 2nd by both methods beats 1st by one
    # and 10th by the other (rrf_k=60: 1/62 + 1/62 > 1/61 + 1/70).
    a = _chunk("a", index=0)
    b = _chunk("b", index=1)
    c = _chunk("c", index=2)
    first = [SearchResult(chunk=a, score=0.9), SearchResult(chunk=b, score=0.8)]
    second = [SearchResult(chunk=b, score=0.9), SearchResult(chunk=c, score=0.8)]
    fused = reciprocal_rank_fusion([first, second])
    assert [r.chunk.text for r in fused] == ["b", "a", "c"]


def test_rrf_scores_fall_with_rank_and_empty_lists_fuse_to_empty():
    a = _chunk("a", index=0)
    b = _chunk("b", index=1)
    fused = reciprocal_rank_fusion(
        [[SearchResult(chunk=a, score=1.0), SearchResult(chunk=b, score=0.5)]]
    )
    assert fused[0].score > fused[1].score
    assert fused[0].chunk.text == "a"
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], k=0)


def test_hybrid_top_results_are_the_ones_both_methods_like():
    # Vector ranks chunk 2 first, BM25 ranks chunk 3 first; both rank
    # each other second. Fusion puts those two on top, ahead of the
    # chunks each method alone ranks low.
    retriever, _ = _hybrid(
        [
            "search engines rank pages by relevance and links",
            "quark flavors up down strange charm bottom top",
            "search search search search search search search search",
            "the quark model explains particle physics and search results",
        ]
    )
    results = retriever.retrieve("quark search", k=2, lambda_mult=1.0)
    assert [r.chunk.index for r in results] == [2, 3]


def test_hybrid_matches_manual_fusion_then_mmr():
    retriever, enc = _hybrid(
        [
            "cats sleep all day in the warm sun",
            "cats cats cats cats cats purr softly while the rain falls "
            "outside and the house is quiet and warm",
            "dogs run in the park every morning",
            "sleep sleep sleep and dream all night long under the stars",
        ]
    )
    query = "cats sleep"
    query_vector = enc.embed(query)
    vector_hits = [
        result
        for result, _ in retriever.store.search_with_vectors(query_vector, k=12)
    ]
    keyword_hits = retriever.bm25.search(query, k=12)
    fused = reciprocal_rank_fusion(
        [vector_hits, keyword_hits], k=retriever.rrf_k
    )
    pairs = [
        (result, retriever.store.vector_for(result.chunk)) for result in fused
    ]
    expected = mmr_rerank(query_vector, pairs, k=4, lambda_mult=0.7)
    assert [
        r.chunk.index for r in retriever.retrieve(query, k=4)
    ] == [r.chunk.index for r in expected]


def test_hybrid_mmr_pushes_near_duplicate_down_after_fusion():
    # With pure fusion (lambda 1) the two cats-heavy chunks sit on top.
    # With the default diversity trade-off, the near-copy of the top
    # chunk drops to last and the other angles come up instead.
    retriever, _ = _hybrid(
        [
            "cats sleep all day in the warm sun",
            "cats cats cats cats cats purr softly while the rain falls "
            "outside and the house is quiet and warm",
            "dogs run in the park every morning",
            "sleep sleep sleep and dream all night long under the stars",
        ]
    )
    fused_only = retriever.retrieve("cats sleep", k=4, lambda_mult=1.0)
    assert [r.chunk.index for r in fused_only] == [1, 0, 3, 2]
    diverse = retriever.retrieve("cats sleep", k=4)
    assert [r.chunk.index for r in diverse] == [1, 3, 2, 0]


def test_hybrid_retrieve_respects_where_and_k():
    enc = HashingEmbedder(dim=256)
    chunks = [
        _chunk("cats sleep all day", doc_id="cats", index=0),
        _chunk("cats nap in the sun", doc_id="cats", index=1),
        _chunk("cats purr on the sofa", doc_id="dogs", index=0),
    ]
    retriever = HybridRetriever.from_chunks(chunks, enc)

    assert len(retriever.retrieve("cats sleep", k=2)) == 2
    filtered = retriever.retrieve("cats sleep", k=5, where={"doc_id": "dogs"})
    assert [r.chunk.doc_id for r in filtered] == ["dogs"]
    assert retriever.retrieve("cats", where={"doc_id": "nope"}) == []


def test_hybrid_empty_store_returns_nothing():
    enc = HashingEmbedder(dim=256)
    retriever = HybridRetriever.from_chunks([], enc)
    assert retriever.retrieve("cats") == []


def test_hybrid_validates_inputs():
    retriever, enc = _hybrid(["cats sleep"])
    with pytest.raises(ValueError):
        retriever.retrieve("cats", k=0)
    with pytest.raises(ValueError):
        retriever.retrieve("cats", lambda_mult=1.5)
    with pytest.raises(ValueError):
        HybridRetriever(retriever.store, enc, retriever.bm25, lambda_mult=2.0)
    with pytest.raises(ValueError):
        HybridRetriever(retriever.store, enc, retriever.bm25, rrf_k=0)
    with pytest.raises(ValueError):
        HybridRetriever(retriever.store, HashingEmbedder(dim=64), retriever.bm25)


def test_vector_for_returns_the_stored_vector():
    enc = HashingEmbedder(dim=256)
    chunks = [_chunk("cats sleep all day", index=0)]
    store = VectorStore.from_chunks(chunks, enc)
    assert store.vector_for(chunks[0]) == enc.embed("cats sleep all day")
    assert store.vector_for(_chunk("not in the store", index=9)) is None


def test_hybrid_over_sample_docs():
    from rag.pipeline import build_chunks

    chunks = build_chunks("data/sample")
    enc = HashingEmbedder(dim=256)
    retriever = HybridRetriever.from_chunks(chunks, enc)

    results = retriever.retrieve("vector search", k=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)

    filtered = retriever.retrieve(
        "vector search", k=5, where={"doc_id": "vector-search.txt"}
    )
    assert filtered
    assert all(r.chunk.doc_id == "vector-search.txt" for r in filtered)
