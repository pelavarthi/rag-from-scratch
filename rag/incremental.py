"""Incremental indexing: keep an index fresh without rebuilding it.

A full rebuild re-embeds every document, even the ones that did not
change. This module remembers what it already indexed — a manifest of
one content hash per document — so an update only touches what moved:

- added: a file not in the manifest gets chunked and embedded
- changed: a file whose hash moved gets its old chunks removed, then
  re-chunked and embedded
- deleted: a file that is gone gets its chunks removed
- unchanged: same hash, skipped entirely — no embedding work at all

The manifest lives next to the index JSON (``<index>.manifest.json``
by default). ``watch`` polls the directory and applies updates as
files change, so the index can catch up underneath a running server —
the Go server picks the new index up with ``POST /reload``.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_document
from .embeddings import Embedder, embed_chunks
from .ingest import TEXT_EXTENSIONS, load_document
from .vector_store import VectorStore


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes. Same content, same hash, every run."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DocState:
    """What the manifest remembers about one indexed document."""

    hash: str
    mtime: float
    chunks: int


@dataclass
class Manifest:
    """One entry per indexed document, keyed by document id."""

    docs: dict[str, DocState] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        """Read a manifest file. A missing file means an empty index."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            docs={
                doc_id: DocState(**state)
                for doc_id, state in data.get("docs", {}).items()
            }
        )

    def save(self, path: str | Path) -> None:
        """Write the manifest atomically, like VectorStore.save."""
        path = Path(path)
        data = {
            "docs": {
                doc_id: {
                    "hash": state.hash,
                    "mtime": state.mtime,
                    "chunks": state.chunks,
                }
                for doc_id, state in self.docs.items()
            }
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)


@dataclass
class FileInfo:
    """What a scan found on disk for one document."""

    path: Path
    hash: str
    mtime: float


def scan_directory(
    root: str | Path, extensions: set[str] | None = None
) -> dict[str, FileInfo]:
    """Hash every supported file under `root`, keyed by document id.

    The id is the path relative to `root`, the same id `load_document`
    assigns, so a scan lines up with the manifest one-to-one.
    """
    root = Path(root)
    extensions = extensions or TEXT_EXTENSIONS
    found: dict[str, FileInfo] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            doc_id = str(path.relative_to(root))
            found[doc_id] = FileInfo(
                path=path,
                hash=file_hash(path),
                mtime=path.stat().st_mtime,
            )
    return found


@dataclass
class Diff:
    """What changed between the manifest and the latest scan."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def diff_manifest(manifest: Manifest, scanned: dict[str, FileInfo]) -> Diff:
    """Sort every document into added / changed / deleted / unchanged.

    A file is changed when its hash moved — the hash, not the mtime,
    is the test, so touching a file without editing it is a no-op.
    """
    diff = Diff()
    for doc_id, info in scanned.items():
        state = manifest.docs.get(doc_id)
        if state is None:
            diff.added.append(doc_id)
        elif state.hash != info.hash:
            diff.changed.append(doc_id)
        else:
            diff.unchanged.append(doc_id)
    for doc_id in manifest.docs:
        if doc_id not in scanned:
            diff.deleted.append(doc_id)
    return diff


@dataclass
class UpdateSummary:
    """What one update did, for logs and for tests."""

    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    embedded: int = 0

    @property
    def moved(self) -> bool:
        """True when anything was added, changed, or deleted."""
        return self.added + self.changed + self.deleted > 0

    def __str__(self) -> str:
        return (
            f"+{self.added} ~{self.changed} -{self.deleted} "
            f"={self.unchanged} unchanged, "
            f"{self.embedded} chunks embedded, "
            f"{self.chunks_removed} removed"
        )


class IncrementalIndexer:
    """Keeps one store in sync with one directory of documents."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        source_dir: str | Path,
        index_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        size: int = 500,
        overlap: int = 50,
    ):
        self.store = store
        self.embedder = embedder
        self.source_dir = Path(source_dir)
        self.index_path = Path(index_path) if index_path else None
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else (self.index_path.with_suffix(".manifest.json") if self.index_path else None)
        )
        self.size = size
        self.overlap = overlap
        self.manifest = (
            Manifest.load(self.manifest_path) if self.manifest_path else Manifest()
        )

    def _index_one(self, doc_id: str, info: FileInfo) -> int:
        """Chunk, embed, and add one document. Returns chunks added."""
        doc = load_document(info.path, root=self.source_dir)
        chunks = chunk_document(doc, size=self.size, overlap=self.overlap)
        self.store.add(embed_chunks(chunks, self.embedder))
        self.manifest.docs[doc_id] = DocState(
            hash=info.hash, mtime=info.mtime, chunks=len(chunks)
        )
        return len(chunks)

    def update(self) -> UpdateSummary:
        """Apply one scan: add, re-index, and delete, then persist.

        The store and manifest are only rewritten when something
        moved, so a quiet update costs one directory scan and nothing
        else.
        """
        scanned = scan_directory(self.source_dir)
        diff = diff_manifest(self.manifest, scanned)
        summary = UpdateSummary(
            added=len(diff.added),
            changed=len(diff.changed),
            deleted=len(diff.deleted),
            unchanged=len(diff.unchanged),
        )

        for doc_id in diff.deleted:
            summary.chunks_removed += self.store.remove_doc(doc_id)
            del self.manifest.docs[doc_id]

        for doc_id in diff.changed:
            summary.chunks_removed += self.store.remove_doc(doc_id)
            summary.chunks_added += self._index_one(doc_id, scanned[doc_id])

        for doc_id in diff.added:
            summary.chunks_added += self._index_one(doc_id, scanned[doc_id])

        summary.embedded = summary.chunks_added
        if summary.moved:
            if self.index_path:
                self.store.save(self.index_path)
            if self.manifest_path:
                self.manifest.save(self.manifest_path)
        return summary

    def watch(
        self,
        interval: float = 1.0,
        on_update: Callable[[UpdateSummary], None] | None = None,
        max_rounds: int | None = None,
    ) -> int:
        """Poll the directory forever, applying updates as files change.

        `on_update` runs only when a round moved something. `max_rounds`
        caps the loop (handy for scripts and tests); None means run
        until interrupted with Ctrl-C. Returns rounds run.
        """
        rounds = 0
        try:
            while max_rounds is None or rounds < max_rounds:
                summary = self.update()
                if summary.moved and on_update:
                    on_update(summary)
                rounds += 1
                if max_rounds is None or rounds < max_rounds:
                    time.sleep(interval)
        except KeyboardInterrupt:
            pass
        return rounds


def open_indexer(
    source_dir: str | Path,
    index_path: str | Path,
    embedder: Embedder,
    size: int = 500,
    overlap: int = 50,
) -> IncrementalIndexer:
    """Load an existing index and manifest, or start fresh.

    The manifest path defaults to ``<index>.manifest.json`` next to the
    index. A first run over a directory with no index yet indexes
    everything, exactly like a full build.
    """
    index_path = Path(index_path)
    store = VectorStore.load(index_path) if index_path.exists() else VectorStore(embedder.dim)
    return IncrementalIndexer(
        store=store,
        embedder=embedder,
        source_dir=source_dir,
        index_path=index_path,
        size=size,
        overlap=overlap,
    )
