"""Turn text into vectors, so search can compare meaning by numbers.

An `Embedder` is anything with a `dim` and an `embed` method. The rest of
the pipeline only talks to that interface, so the baseline encoder below
can be swapped for a real model later without touching the callers.

The baseline is a hashing encoder: each word is hashed into one of `dim`
buckets and counted, then the vector is scaled to unit length. Two texts
that share words land close together; two with nothing in common land
far apart. It knows nothing about synonyms, but it is deterministic,
needs no downloads, and is enough to build search on top of.
"""

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .document import Chunk

_WORD = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    @property
    def dim(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed one piece of text."""
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Defaults to calling `embed` for each text."""
        return [self.embed(t) for t in texts]


def normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length. A zero vector stays zero."""
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        return vector
    return [x / length for x in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How alike two vectors are: 1 for identical, 0 for unrelated.

    Assumes unit-length vectors (what every embedder here returns), so
    this is just the dot product. Raises if the lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


class HashingEmbedder:
    """Baseline encoder: hashed bag of words, standard library only.

    Words are lowercased, hashed with MD5 (stable across runs, unlike
    Python's built-in hash), and counted into `dim` buckets. The result
    is L2-normalized, so comparing two texts is just a dot product.
    """

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in _WORD.findall(text.lower()):
            bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4]) % self._dim
            vector[bucket] += 1.0
        return normalize(vector)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@dataclass
class EmbeddedChunk:
    """A chunk paired with its vector, ready to be searched."""

    chunk: Chunk
    vector: list[float]


def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> list[EmbeddedChunk]:
    """Embed every chunk with `embedder`, keeping chunk and vector together."""
    vectors = embedder.embed_many([c.text for c in chunks])
    return [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)]
