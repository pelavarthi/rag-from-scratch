"""Tests for incremental indexing: add, change, delete, watch."""

from rag.document import Chunk
from rag.embeddings import HashingEmbedder
from rag.incremental import (
    DocState,
    IncrementalIndexer,
    Manifest,
    UpdateSummary,
    diff_manifest,
    file_hash,
    open_indexer,
    scan_directory,
)
from rag.vector_store import VectorStore


class CountingEmbedder(HashingEmbedder):
    """HashingEmbedder that counts how many chunks it embedded."""

    def __init__(self, dim=64):
        super().__init__(dim=dim)
        self.embedded = 0

    def embed_many(self, texts):
        self.embedded += len(texts)
        return super().embed_many(texts)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _indexer(tmp_path, dim=64):
    enc = CountingEmbedder(dim=dim)
    store = VectorStore(dim=dim)
    return IncrementalIndexer(
        store=store,
        embedder=enc,
        source_dir=tmp_path / "docs",
        index_path=tmp_path / "index.json",
    ), enc


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "a.txt"
    _write(path, "cats sleep")
    first = file_hash(path)
    _write(path, "cats sleep all day")
    assert file_hash(path) != first
    _write(path, "cats sleep")
    assert file_hash(path) == first


def test_diff_sorts_added_changed_deleted_unchanged(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "keep.txt", "keep me")
    _write(docs / "edit.txt", "old text here")
    _write(docs / "new.txt", "brand new doc")
    manifest = Manifest()
    manifest.docs["keep.txt"] = DocState(
        hash=file_hash(docs / "keep.txt"), mtime=0, chunks=1
    )
    manifest.docs["edit.txt"] = DocState(hash="deadbeef", mtime=0, chunks=1)
    manifest.docs["gone.txt"] = DocState(hash="deadbeef", mtime=0, chunks=1)
    diff = diff_manifest(manifest, scan_directory(docs))
    assert diff.added == ["new.txt"]
    assert diff.changed == ["edit.txt"]
    assert diff.deleted == ["gone.txt"]
    assert diff.unchanged == ["keep.txt"]


def test_touching_a_file_without_editing_is_unchanged(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "a.txt", "cats sleep all day in the sun")
    indexer, enc = _indexer(tmp_path)
    first = indexer.update()
    assert first.added == 1 and first.embedded == len(indexer.store)
    # Rewrite identical content: same hash, so nothing re-embeds.
    _write(docs / "a.txt", "cats sleep all day in the sun")
    second = indexer.update()
    assert second == UpdateSummary(unchanged=1)
    assert enc.embedded == first.embedded


def test_update_adds_changes_and_deletes(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "cats.txt", "cats sleep all day in the sun")
    _write(docs / "dogs.txt", "dogs run in the park every morning")
    indexer, enc = _indexer(tmp_path)
    indexer.update()
    assert len(indexer.store) == 2
    assert indexer.store.search_text("cats nap", enc)[0].chunk.text.startswith("cats")

    # Edit one doc: its old chunk goes, a new one arrives, the other stays.
    _write(docs / "cats.txt", "quantum physics lectures on monday")
    summary = indexer.update()
    assert (summary.added, summary.changed, summary.deleted) == (0, 1, 0)
    assert summary.chunks_removed == 1 and summary.chunks_added == 1
    assert len(indexer.store) == 2
    assert indexer.store.search_text("quantum lectures", enc)[0].chunk.text.startswith("quantum")
    assert not indexer.store.search_text("cats nap", enc)[0].chunk.text.startswith("cats")

    # Delete a doc: its chunks leave the store.
    (docs / "dogs.txt").unlink()
    summary = indexer.update()
    assert (summary.added, summary.changed, summary.deleted) == (0, 0, 1)
    assert len(indexer.store) == 1
    assert indexer.store.search_text("dogs run", enc)[0].chunk.doc_id == "cats.txt"

    # Add a doc: only the new one is embedded.
    embedded_before = enc.embedded
    _write(docs / "birds.txt", "birds fly over the lake at dawn")
    summary = indexer.update()
    assert (summary.added, summary.changed, summary.deleted) == (1, 0, 0)
    assert enc.embedded - embedded_before == 1
    assert len(indexer.store) == 2


def test_incremental_matches_full_build(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "a.txt", "cats sleep all day in the sun")
    _write(docs / "b.txt", "dogs run in the park every morning")
    indexer, enc = _indexer(tmp_path)
    indexer.update()
    _write(docs / "a.txt", "quantum physics lectures on monday")
    _write(docs / "c.txt", "birds fly over the lake at dawn")
    (docs / "b.txt").unlink()
    indexer.update()

    fresh = VectorStore.from_chunks(
        [
            Chunk(doc_id="a.txt", text="quantum physics lectures on monday", index=0, start_char=0, end_char=len("quantum physics lectures on monday")),
            Chunk(doc_id="c.txt", text="birds fly over the lake at dawn", index=0, start_char=0, end_char=len("birds fly over the lake at dawn")),
        ],
        HashingEmbedder(dim=64),
    )
    query = enc.embed("quantum lectures")
    assert [r.chunk.text for r in indexer.store.search(query)] == [
        r.chunk.text for r in fresh.search(query)
    ]
    assert [r.score for r in indexer.store.search(query)] == [
        r.score for r in fresh.search(query)
    ]


def test_manifest_survives_reload(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "a.txt", "cats sleep all day in the sun")
    indexer, enc = _indexer(tmp_path)
    indexer.update()
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "index.manifest.json").exists()

    # A new indexer over the same files finds nothing to do.
    reopened = open_indexer(docs, tmp_path / "index.json", enc)
    assert reopened.update() == UpdateSummary(unchanged=1)
    assert enc.embedded == 1


def test_remove_doc_only_removes_that_doc():
    store = VectorStore(dim=64)
    enc = HashingEmbedder(dim=64)
    chunks = [
        Chunk(doc_id="a", text="cats sleep", index=0, start_char=0, end_char=10),
        Chunk(doc_id="a", text="cats nap", index=1, start_char=10, end_char=18),
        Chunk(doc_id="b", text="dogs run", index=0, start_char=0, end_char=8),
    ]
    from rag.embeddings import embed_chunks

    store.add(embed_chunks(chunks, enc))
    assert store.remove_doc("a") == 2
    assert len(store) == 1
    assert store.remove_doc("missing") == 0
    assert len(store) == 1


def test_watch_applies_changes_between_rounds(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "a.txt", "cats sleep all day in the sun")
    indexer, enc = _indexer(tmp_path)
    seen: list[UpdateSummary] = []

    def on_update(summary):
        seen.append(summary)

    # Change the doc after the first round, from inside the callback.
    def changing_callback(summary):
        on_update(summary)
        if len(seen) == 1:
            _write(docs / "a.txt", "quantum physics lectures on monday")

    indexer.watch(interval=0, on_update=changing_callback, max_rounds=3)
    assert len(seen) == 2
    assert seen[0].added == 1 and seen[1].changed == 1
    assert indexer.store.search_text("quantum lectures", enc)[0].chunk.text.startswith("quantum")


def test_watch_quiet_rounds_do_not_call_back(tmp_path):
    docs = tmp_path / "docs"
    _write(docs / "a.txt", "cats sleep all day in the sun")
    indexer, _ = _indexer(tmp_path)
    indexer.update()
    seen: list[UpdateSummary] = []
    indexer.watch(interval=0, on_update=seen.append, max_rounds=2)
    assert seen == []
