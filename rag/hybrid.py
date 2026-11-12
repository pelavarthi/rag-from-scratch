"""Hybrid search: keyword search plus vector search, fused into one ranking.

Vector search and keyword search fail in different ways. Vector search
understands that "car" and "automobile" are related, but it dilutes a
rare exact word: a long chunk that mentions "quark" once can score lower
than a short chunk that shares only common words with the question.
Keyword search has the opposite blind spot: it only sees exact words, so
"car" never matches "automobile", and a common word repeated ten times
can outscore a rare word that actually identifies the chunk.

Hybrid search runs both and merges the two rankings. The merge is
reciprocal rank fusion (RRF): a chunk's score is the sum of
`1 / (rrf_k + rank)` over each list it appears in, so only the *rank*
matters, never the raw scores — which is what makes the merge fair,
because a BM25 score and a cosine similarity live on different scales.
A chunk ranked 2nd by both methods beats a chunk ranked 1st by one and
10th by the other. The fused list is then passed through MMR (Day 4),
so the final results are relevant, cover different angles, and include
chunks either method alone would have missed.
"""

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .document import Chunk
from .embeddings import Embedder
from .retrieval import _as_predicate, mmr_rerank
from .vector_store import SearchResult, VectorStore

_WORD = re.compile(r"\w+")
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words, the same split the embedder uses."""
    return _WORD.findall(text.lower())


def _chunk_key(chunk: Chunk) -> tuple[str, int]:
    """Identity of a chunk: its document plus its position in it."""
    return (chunk.doc_id, chunk.index)


class BM25Index:
    """Keyword search over chunks, scored by BM25.

    For each query word, a chunk earns points for how often the word
    appears in it (term frequency), scaled by how rare the word is
    across all chunks (inverse document frequency), and damped by how
    long the chunk is relative to the average — so a short chunk that
    mentions the word once can beat a long chunk that buries it.
    `k1` caps the reward for repeating a word; `b` sets how much
    length matters (0 ignores length, 1 fully normalizes by it).
    """

    def __init__(self, k1: float = _DEFAULT_K1, b: float = _DEFAULT_B):
        if k1 < 0:
            raise ValueError("k1 must be at least 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter] = []
        self._doc_freq: Counter = Counter()
        self._doc_lengths: list[int] = []
        self._avg_len = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> "BM25Index":
        """Build an index over `chunks` in one call."""
        index = cls(k1=k1, b=b)
        index._chunks = list(chunks)
        for chunk in chunks:
            words = tokenize(chunk.text)
            freqs = Counter(words)
            index._term_freqs.append(freqs)
            index._doc_lengths.append(len(words))
            for word in freqs:
                index._doc_freq[word] += 1
        if chunks:
            index._avg_len = sum(index._doc_lengths) / len(chunks)
        return index

    def _idf(self, word: str) -> float:
        """How rare `word` is: high when few chunks contain it."""
        n = len(self._chunks)
        df = self._doc_freq.get(word, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _score(self, query_words: list[str], doc: int) -> float:
        """BM25 score of chunk `doc` for the query words."""
        freqs = self._term_freqs[doc]
        length = self._doc_lengths[doc]
        total = 0.0
        for word in set(query_words):
            tf = freqs.get(word, 0)
            if tf == 0:
                continue
            norm = 1 - self._b + self._b * (length / self._avg_len)
            total += self._idf(word) * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        return total

    def search(
        self,
        query: str,
        k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[SearchResult]:
        """Return the `k` chunks whose words best match `query`, best first.

        `where` is an optional predicate: only chunks it accepts are
        scored. Chunks sharing no word with the query score zero and
        are left out. Scores are raw BM25 scores, not 0-to-1, so they
        are only comparable within this index — RRF fuses by rank.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        words = tokenize(query)
        if not words or not self._chunks:
            return []
        scored = []
        for i, chunk in enumerate(self._chunks):
            if where is not None and not where(chunk):
                continue
            score = self._score(words, i)
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """Merge ranked lists into one ranking, by rank only.

    Each chunk's fused score is the sum of `1 / (k + rank)` over every
    list it appears in, with rank starting at 1. `k` (60 by convention)
    dampens the gap between ranks: a chunk 1st in one list and 10th in
    another still outscores a chunk 1st in one list and absent from the
    other. The returned `SearchResult` scores are fusion scores, not
    similarities, and only their order matters.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[tuple[str, int], tuple[SearchResult, float]] = {}
    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            key = _chunk_key(result.chunk)
            score = 1.0 / (k + rank)
            if key in fused:
                first, total = fused[key]
                fused[key] = (first, total + score)
            else:
                fused[key] = (result, score)
    ranked = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    return [SearchResult(chunk=result.chunk, score=score) for result, score in ranked]


@dataclass
class HybridRetriever:
    """Search by meaning and by exact words at once, then rerank.

    Runs vector search and BM25 over the same chunks, fuses the two
    rankings with RRF, and reranks the fused list with MMR, so the
    final `k` results are relevant, diverse, and reachable by either
    kind of match. `rrf_k` is passed through to the fusion; the other
    settings match `Retriever` (see `rag/retrieval.py`).
    """

    store: VectorStore
    embedder: Embedder
    bm25: BM25Index
    lambda_mult: float = 0.7
    fetch_multiplier: int = 3
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.embedder.dim != self.store.dim:
            raise ValueError(
                f"embedder dim {self.embedder.dim} does not match "
                f"store dim {self.store.dim}"
            )
        if not 0 <= self.lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        if self.fetch_multiplier < 1:
            raise ValueError("fetch_multiplier must be at least 1")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        **kwargs,
    ) -> "HybridRetriever":
        """Build the vector store and the BM25 index over `chunks`."""
        store = VectorStore.from_chunks(chunks, embedder)
        bm25 = BM25Index.from_chunks(chunks)
        return cls(store=store, embedder=embedder, bm25=bm25, **kwargs)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: dict | Callable[[Chunk], bool] | None = None,
        lambda_mult: float | None = None,
        fetch_k: int | None = None,
    ) -> list[SearchResult]:
        """Return the `k` most useful chunks for `query`, best first.

        `where` narrows the search before ranking: a dict of metadata
        conditions or a predicate on the chunk, applied to both indexes.
        `lambda_mult` overrides the diversity trade-off for this query;
        `fetch_k` overrides how many candidates each index contributes
        and is raised to `k` when smaller.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        lam = self.lambda_mult if lambda_mult is None else lambda_mult
        if not 0 <= lam <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        if fetch_k is None:
            fetch_k = k * self.fetch_multiplier
        fetch_k = max(fetch_k, k)

        predicate = _as_predicate(where)
        query_vector = self.embedder.embed(query)
        vector_hits = [
            result
            for result, _ in self.store.search_with_vectors(
                query_vector, k=fetch_k, where=predicate
            )
        ]
        keyword_hits = self.bm25.search(query, k=fetch_k, where=predicate)
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=self.rrf_k)
        if not fused:
            return []

        # MMR needs each candidate's vector; if one is missing (the
        # retriever was built from mismatched indexes), fall back to
        # the fused order instead of guessing.
        pairs = []
        for result in fused:
            vector = self.store.vector_for(result.chunk)
            if vector is None:
                return fused[:k]
            pairs.append((result, vector))
        return mmr_rerank(query_vector, pairs, k=k, lambda_mult=lam)
