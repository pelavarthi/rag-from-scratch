"""Retrieve the right chunks for a question.

Plain top-k search has two weaknesses. First, it returns the nearest
chunks even when they all say the same thing, so the answer gets five
copies of one fact and nothing else. Second, it cannot narrow the search
to one document or one kind of source. This module fixes both:

- metadata filters narrow the search before ranking, so a question about
  one document only ever sees that document's chunks;
- MMR (maximal marginal relevance) reranks the top candidates so each new
  pick adds something the earlier picks did not already say.

`Retriever` ties the pieces together: it embeds the question, fetches
more candidates than it needs, and returns the `k` most useful ones.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .document import Chunk
from .embeddings import Embedder, cosine_similarity
from .vector_store import SearchResult, VectorStore

_MISSING = object()


def matches_metadata(chunk: Chunk, conditions: dict) -> bool:
    """True when `chunk` satisfies every condition in `conditions`.

    Each key names a field to check: `"doc_id"` matches the chunk's
    document id, any other key matches the chunk's metadata. A value is
    either one expected value, or a list of acceptable values (any of
    them matches). A chunk missing the key does not match.
    """
    for key, expected in conditions.items():
        actual = chunk.doc_id if key == "doc_id" else chunk.metadata.get(key, _MISSING)
        if actual is _MISSING:
            return False
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def mmr_rerank(
    query_vector: list[float],
    candidates: list[tuple[SearchResult, list[float]]],
    k: int,
    lambda_mult: float = 0.7,
) -> list[SearchResult]:
    """Pick `k` results from `candidates` by maximal marginal relevance.

    `candidates` are (result, vector) pairs, best first by score. The
    first pick is always the most relevant chunk. Each later pick is the
    candidate that best balances relevance to the query against novelty:
    `lambda_mult * relevance - (1 - lambda_mult) * similarity to the
    most similar chunk already picked`. With `lambda_mult` at 1 the
    result is plain top-k; at 0 it is pure diversity.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    if not candidates:
        return []

    picked = [candidates[0]]
    rest = list(candidates[1:])
    while rest and len(picked) < k:
        def mmr_score(item: tuple[SearchResult, list[float]]) -> float:
            result, vector = item
            overlap = max(
                cosine_similarity(vector, other_vector)
                for _, other_vector in picked
            )
            return lambda_mult * result.score - (1 - lambda_mult) * overlap

        best = max(range(len(rest)), key=lambda i: mmr_score(rest[i]))
        picked.append(rest.pop(best))
    return [result for result, _ in picked]


def _as_predicate(
    where: dict | Callable[[Chunk], bool] | None,
) -> Callable[[Chunk], bool] | None:
    """Turn a `where` argument into a chunk predicate, or pass it through."""
    if where is None or callable(where):
        return where
    if isinstance(where, dict):
        return lambda chunk: matches_metadata(chunk, where)
    raise TypeError("where must be a dict of metadata conditions or a callable")


@dataclass
class Retriever:
    """Search a store for a question: embed, over-fetch, rerank.

    `lambda_mult` trades relevance against diversity (see `mmr_rerank`).
    `fetch_multiplier` sets how many candidates are fetched per result
    requested, so MMR has room to choose from; fetching more costs a
    longer scan, so 3 is a reasonable default.
    """

    store: VectorStore
    embedder: Embedder
    lambda_mult: float = 0.7
    fetch_multiplier: int = 3

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
        conditions (see `matches_metadata`) or a predicate on the chunk.
        `lambda_mult` overrides the retriever's default for this query.
        `fetch_k` overrides how many candidates MMR chooses from; it is
        raised to `k` when smaller, since fewer candidates than results
        makes no sense.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        lam = self.lambda_mult if lambda_mult is None else lambda_mult
        if not 0 <= lam <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        if fetch_k is None:
            fetch_k = k * self.fetch_multiplier
        fetch_k = max(fetch_k, k)

        query_vector = self.embedder.embed(query)
        candidates = self.store.search_with_vectors(
            query_vector, k=fetch_k, where=_as_predicate(where)
        )
        return mmr_rerank(query_vector, candidates, k=k, lambda_mult=lam)
