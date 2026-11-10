"""Store embedded chunks and find the nearest ones to a query.

The store is deliberately simple: every embedded chunk lives in a list,
and a search compares the query vector against all of them with cosine
similarity. That brute-force scan is fast enough for a few thousand
chunks, and it is the baseline the approximate index on Day 10 has to
beat. Persistence is one JSON file, so an index can be built once and
reused without re-embedding everything.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .document import Chunk
from .embeddings import EmbeddedChunk, Embedder, cosine_similarity


@dataclass
class SearchResult:
    """One hit: the chunk that matched, and how close it was.

    `score` is the cosine similarity, 1 for an identical direction and
    0 for unrelated text.
    """

    chunk: Chunk
    score: float


class VectorStore:
    """An in-memory index of embedded chunks, searchable by vector."""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._entries: list[EmbeddedChunk] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorStore":
        """Embed every chunk and return a store holding them."""
        store = cls(embedder.dim)
        vectors = embedder.embed_many([c.text for c in chunks])
        store.add(
            [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)]
        )
        return store

    def add(self, embedded: list[EmbeddedChunk]) -> None:
        """Add embedded chunks. Every vector must match the store's dim."""
        for item in embedded:
            if len(item.vector) != self._dim:
                raise ValueError(
                    f"vector dim {len(item.vector)} does not match store dim {self._dim}"
                )
        self._entries.extend(embedded)

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[SearchResult]:
        """Return the `k` chunks closest to `query_vector`, best first.

        `where` is an optional predicate: only chunks it accepts are
        scored, so a search can be narrowed to one document or source
        before ranking.
        """
        return [
            result
            for result, _ in self.search_with_vectors(query_vector, k=k, where=where)
        ]

    def search_with_vectors(
        self,
        query_vector: list[float],
        k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[SearchResult, list[float]]]:
        """Like `search`, but each hit also carries its vector.

        Rerankers such as MMR need the vectors to compare candidates
        against each other, so they use this instead of `search`.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if len(query_vector) != self._dim:
            raise ValueError(
                f"query dim {len(query_vector)} does not match store dim {self._dim}"
            )
        scored = [
            (
                SearchResult(
                    chunk=item.chunk,
                    score=cosine_similarity(query_vector, item.vector),
                ),
                item.vector,
            )
            for item in self._entries
            if where is None or where(item.chunk)
        ]
        scored.sort(key=lambda pair: pair[0].score, reverse=True)
        return scored[:k]

    def search_text(
        self,
        query: str,
        embedder: Embedder,
        k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[SearchResult]:
        """Embed `query` with `embedder`, then search for the nearest chunks."""
        if embedder.dim != self._dim:
            raise ValueError(
                f"embedder dim {embedder.dim} does not match store dim {self._dim}"
            )
        return self.search(embedder.embed(query), k=k, where=where)

    def save(self, path: str | Path) -> None:
        """Write the whole index to one JSON file, atomically.

        The file holds the dim plus every chunk and its vector, so a
        loaded store searches exactly like the one that was saved.
        """
        path = Path(path)
        data = {
            "dim": self._dim,
            "entries": [
                {"chunk": item.chunk.__dict__, "vector": item.vector}
                for item in self._entries
            ],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Read back a store written by `save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        store = cls(data["dim"])
        store.add(
            [
                EmbeddedChunk(chunk=Chunk(**entry["chunk"]), vector=entry["vector"])
                for entry in data["entries"]
            ]
        )
        return store
