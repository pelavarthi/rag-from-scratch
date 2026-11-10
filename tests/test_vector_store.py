"""Tests for the in-memory vector store: search, ranking, persistence."""

import pytest

from rag.document import Chunk
from rag.embeddings import HashingEmbedder, embed_chunks
from rag.vector_store import SearchResult, VectorStore


def _chunk(text, doc_id="d1", index=0):
    return Chunk(doc_id=doc_id, text=text, index=index, start_char=0, end_char=len(text))


def _store_with(texts, dim=256):
    enc = HashingEmbedder(dim=dim)
    chunks = [_chunk(t, index=i) for i, t in enumerate(texts)]
    return VectorStore.from_chunks(chunks, enc), enc


def test_add_and_len():
    store = VectorStore(dim=64)
    assert len(store) == 0
    enc = HashingEmbedder(dim=64)
    store.add(embed_chunks([_chunk("cats sleep"), _chunk("dogs run")], enc))
    assert len(store) == 2


def test_add_rejects_wrong_dim():
    store = VectorStore(dim=64)
    enc = HashingEmbedder(dim=32)
    with pytest.raises(ValueError):
        store.add(embed_chunks([_chunk("cats sleep")], enc))


def test_invalid_dim_rejected():
    with pytest.raises(ValueError):
        VectorStore(dim=0)


def test_search_ranks_most_similar_first():
    store, enc = _store_with(
        [
            "cats sleep all day in the sun",
            "quantum physics lectures on monday",
            "dogs run in the park every morning",
        ]
    )
    results = store.search(enc.embed("cats nap in the sun all day"))
    assert results[0].chunk.text.startswith("cats sleep")
    assert results[0].score > results[1].score > results[2].score
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_returns_at_most_k():
    store, enc = _store_with(["cats sleep", "dogs run", "birds fly"])
    assert len(store.search(enc.embed("cats"), k=2)) == 2
    assert len(store.search(enc.embed("cats"), k=10)) == 3


def test_search_rejects_bad_k_and_wrong_dim():
    store, enc = _store_with(["cats sleep"])
    with pytest.raises(ValueError):
        store.search(enc.embed("cats"), k=0)
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=1)


def test_search_empty_store_returns_nothing():
    store = VectorStore(dim=256)
    enc = HashingEmbedder(dim=256)
    assert store.search(enc.embed("cats")) == []


def test_search_text_embeds_the_query():
    store, enc = _store_with(
        ["cats sleep all day in the sun", "quantum physics lectures on monday"]
    )
    results = store.search_text("cats nap in the sun all day", enc)
    assert results[0].chunk.text.startswith("cats sleep")


def test_search_text_rejects_mismatched_embedder():
    store, _ = _store_with(["cats sleep"], dim=256)
    with pytest.raises(ValueError):
        store.search_text("cats", HashingEmbedder(dim=64))


def test_zero_vector_query_scores_zero():
    store, enc = _store_with(["cats sleep", "dogs run"])
    results = store.search([0.0] * 256)
    assert all(r.score == 0.0 for r in results)


def test_save_and_load_round_trip(tmp_path):
    store, enc = _store_with(
        [
            "cats sleep all day in the sun",
            "quantum physics lectures on monday",
            "dogs run in the park every morning",
        ]
    )
    path = tmp_path / "index.json"
    store.save(path)

    loaded = VectorStore.load(path)
    assert loaded.dim == store.dim
    assert len(loaded) == 3
    assert [e.chunk.text for e in loaded._entries] == [
        "cats sleep all day in the sun",
        "quantum physics lectures on monday",
        "dogs run in the park every morning",
    ]
    assert loaded._entries[0].chunk.doc_id == "d1"
    assert loaded._entries[0].chunk.index == 0

    before = store.search(enc.embed("cats nap in the sun all day"))
    after = loaded.search(enc.embed("cats nap in the sun all day"))
    assert [r.chunk.text for r in after] == [r.chunk.text for r in before]
    assert [r.score for r in after] == [r.score for r in before]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope.json")


def test_from_chunks_builds_searchable_store():
    chunks = [_chunk("cats sleep all day"), _chunk("quantum physics lectures")]
    store = VectorStore.from_chunks(chunks, HashingEmbedder(dim=128))
    assert len(store) == 2
    assert store.dim == 128
    results = store.search_text("cats nap all day", HashingEmbedder(dim=128))
    assert results[0].chunk.text.startswith("cats sleep")


def test_saved_file_is_plain_json(tmp_path):
    store, _ = _store_with(["cats sleep"])
    path = tmp_path / "index.json"
    store.save(path)
    text = path.read_text(encoding="utf-8")
    assert '"dim": 256' in text
    assert "cats sleep" in text
