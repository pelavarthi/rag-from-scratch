"""Chunking v2: recursive splitting, semantic splitting, and dedup.

Day 1's chunker packs whole sentences up to a size limit. That works,
but it has two blind spots: it ignores document structure (a paragraph
break is a stronger boundary than a sentence break), and it splits on
length alone, so one chunk can straddle two topics while the next one
repeats the same topic. This module adds three tools for that:

- recursive splitting: split on paragraph breaks first, then lines,
  then sentences, then words, so a chunk only crosses a strong boundary
  when it has to.
- semantic splitting: group sentences by how alike they are (cosine
  similarity of their embeddings), so a chunk ends when the topic
  changes instead of when a counter runs out.
- dedup: drop exact copies and near-copies, so the same passage does
  not take up two slots in the index.
"""

import hashlib
import re

from .chunking import split_sentences
from .document import Chunk, Document
from .embeddings import Embedder, cosine_similarity

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]
_WHITESPACE = re.compile(r"\s+")


def _split_keep(text: str, sep: str, base: int) -> list[tuple[str, int, int]]:
    """Split `text` on `sep`, keeping the separator on the piece before it.

    Returns (piece_text, start_char, end_char) triples, with offsets
    relative to the document (`base` is where `text` starts in it).
    """
    pieces: list[tuple[str, int, int]] = []
    start = 0
    while True:
        hit = text.find(sep, start)
        if hit == -1:
            break
        end = hit + len(sep)
        pieces.append((text[start:end], base + start, base + end))
        start = end
    if start < len(text) or not pieces:
        pieces.append((text[start:], base + start, base + len(text)))
    return pieces


def _recursive_pieces(
    text: str, size: int, separators: list[str], base: int = 0
) -> list[tuple[str, int, int]]:
    """Break `text` into leaf pieces no longer than `size`.

    Splits on the first separator that actually divides the text, then
    recurses into any piece still too long. A piece with no separator
    left in it (one very long word) is cut to size.
    """
    if len(text) <= size:
        return [(text, base, base + len(text))]
    if not separators:
        pieces = []
        for i in range(0, len(text), size):
            pieces.append((text[i : i + size], base + i, base + min(i + size, len(text))))
        return pieces

    parts = _split_keep(text, separators[0], base)
    if len(parts) == 1:
        return _recursive_pieces(text, size, separators[1:], base)
    out: list[tuple[str, int, int]] = []
    for piece, start, end in parts:
        out.extend(_recursive_pieces(piece, size, separators, start))
    return out


def _pack(
    pieces: list[tuple[str, int, int]], size: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Greedily pack leaf pieces into chunks of at most `size` chars.

    The next chunk restarts `overlap` characters before the previous one
    ended, so a fact straddling a boundary appears in both.
    """
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(pieces):
        end = start
        length = 0
        while end < len(pieces):
            piece_len = len(pieces[end][0])
            extra = piece_len if end == start else piece_len
            if end > start and length + extra > size:
                break
            length += extra
            end += 1
        if end == start:  # one piece longer than the whole chunk
            end = start + 1

        raw = "".join(p[0] for p in pieces[start:end])
        stripped = raw.strip()
        lead = len(raw) - len(raw.lstrip())
        chunk_start = pieces[start][1] + lead
        chunks.append((stripped, chunk_start, chunk_start + len(stripped)))

        if end >= len(pieces):
            break
        # Restart at the first piece starting at or after
        # (chunk_end - overlap), so the next chunk re-covers the tail of
        # this one. With overlap 0 that is the first piece after this
        # chunk, so nothing repeats.
        target = chunks[-1][2] - overlap
        nxt = end
        for i in range(start + 1, end):
            if pieces[i][1] >= target:
                nxt = i
                break
        start = nxt
    return chunks


def split_recursive(
    text: str,
    size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[tuple[str, int, int]]:
    """Split on structure first, length second.

    Returns (chunk_text, start_char, end_char) triples. Paragraph breaks
    are tried before line breaks, lines before sentences, sentences
    before words, so a chunk keeps a paragraph together when it fits.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size")

    text = text.strip()
    if not text:
        return []

    separators = _DEFAULT_SEPARATORS if separators is None else separators
    pieces = _recursive_pieces(text, size, separators)
    return _pack(pieces, size, overlap)


def split_semantic(
    text: str,
    embedder: Embedder,
    threshold: float = 0.5,
    max_size: int = 1000,
) -> list[tuple[str, int, int]]:
    """Group sentences by topic instead of by length.

    Each sentence is embedded, and a chunk keeps growing while each new
    sentence is similar enough (cosine similarity >= `threshold`) to the
    one before it. When the similarity drops, the topic has changed, so
    a new chunk starts. `max_size` caps a chunk that never changes
    topic. Returns (chunk_text, start_char, end_char) triples.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    text = text.strip()
    if not text:
        return []

    sentences = split_sentences(text)
    offsets: list[int] = []
    cursor = 0
    for sentence in sentences:
        cursor = text.index(sentence, cursor)
        offsets.append(cursor)
        cursor += len(sentence)

    vectors = embedder.embed_many(sentences)

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(sentences):
        end = start + 1
        length = len(sentences[start])
        while end < len(sentences):
            similar = cosine_similarity(vectors[end - 1], vectors[end]) >= threshold
            if not similar:
                break
            extra = len(sentences[end]) + 1
            if length + extra > max_size:
                break
            length += extra
            end += 1

        chunk_start = offsets[start]
        chunk_end = offsets[end - 1] + len(sentences[end - 1])
        chunks.append((" ".join(sentences[start:end]), chunk_start, chunk_end))
        start = end
    return chunks


def normalize_text(text: str) -> str:
    """One canonical form for a chunk's text: lowercase, single spaces."""
    return _WHITESPACE.sub(" ", text.strip().lower())


def text_fingerprint(text: str) -> str:
    """A short hash of the normalized text; equal text, equal hash."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()[:16]


def deduplicate_chunks(
    chunks: list[Chunk],
    embedder: Embedder | None = None,
    similarity_threshold: float = 0.97,
) -> list[Chunk]:
    """Drop exact copies and near-copies, keeping the first of each.

    Exact copies are caught by fingerprint, no embedder needed. When an
    `embedder` is given, a chunk whose vector is nearly identical
    (cosine >= `similarity_threshold`) to one already kept is also
    dropped — that catches reworded repeats and overlapping windows
    that say the same thing twice.
    """
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    kept: list[Chunk] = []
    seen: set[str] = set()
    kept_vectors: list[list[float]] = []
    vectors = embedder.embed_many([c.text for c in chunks]) if embedder else None

    for i, chunk in enumerate(chunks):
        fingerprint = text_fingerprint(chunk.text)
        if fingerprint in seen:
            continue
        if vectors is not None and any(
            cosine_similarity(vectors[i], kept_v) >= similarity_threshold
            for kept_v in kept_vectors
        ):
            continue
        seen.add(fingerprint)
        kept.append(chunk)
        if vectors is not None:
            kept_vectors.append(vectors[i])
    return kept


def chunk_document_v2(
    doc: Document,
    strategy: str = "recursive",
    size: int = 500,
    overlap: int = 50,
    embedder: Embedder | None = None,
    threshold: float = 0.5,
) -> list[Chunk]:
    """Chunk one document with the v2 strategy, tagging each chunk.

    `strategy` is "recursive" (structure first) or "semantic" (topic
    first, needs an embedder). Chunks carry "strategy" in their
    metadata so the index knows how they were made.
    """
    if strategy == "recursive":
        triples = split_recursive(doc.text, size=size, overlap=overlap)
    elif strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        triples = split_semantic(doc.text, embedder, threshold=threshold, max_size=size * 2)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    out = []
    for i, (text, start, end) in enumerate(triples):
        out.append(
            Chunk(
                doc_id=doc.id,
                text=text,
                index=i,
                start_char=start,
                end_char=end,
                metadata={
                    "title": doc.title,
                    "source": doc.source,
                    "strategy": strategy,
                },
            )
        )
    return out
