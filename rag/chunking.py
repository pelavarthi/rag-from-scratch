"""Split document text into overlapping, sentence-aware chunks.

The idea is simple: a chunk should be big enough to answer a question on
its own, small enough to fit in a prompt, and never cut a sentence in half
unless the sentence itself is longer than the chunk.
"""

import re

from .document import Chunk, Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping the punctuation."""
    parts = _SENTENCE_END.split(text.strip())
    return [p for p in parts if p]


def split_into_chunks(
    text: str, size: int = 500, overlap: int = 50
) -> list[tuple[str, int, int]]:
    """Return (chunk_text, start_char, end_char) triples.

    Sentences are packed greedily until adding another would pass `size`
    characters. The next chunk restarts `overlap` characters before the
    previous one ended, so a fact straddling a boundary appears in both.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size")

    text = text.strip()
    if not text:
        return []

    chunks: list[tuple[str, int, int]] = []
    sentences = split_sentences(text)
    # Character offset of each sentence in the original text.
    offsets: list[int] = []
    cursor = 0
    for sentence in sentences:
        cursor = text.index(sentence, cursor)
        offsets.append(cursor)
        cursor += len(sentence)

    start = 0  # index of the first sentence in the current chunk
    while start < len(sentences):
        end = start
        length = 0
        while end < len(sentences):
            extra = len(sentences[end]) + (1 if end > start else 0)
            if end > start and length + extra > size:
                break
            length += extra
            end += 1
        if end == start:  # one sentence longer than the whole chunk
            end = start + 1

        chunk_start = offsets[start]
        chunk_end = offsets[end - 1] + len(sentences[end - 1])
        chunks.append((" ".join(sentences[start:end]), chunk_start, chunk_end))

        if end >= len(sentences):
            break
        # Restart at the sentence containing (chunk_end - overlap), so the
        # next chunk re-covers the tail of this one. Always move forward.
        target = chunk_end - overlap
        nxt = start + 1
        for i in range(start + 1, end):
            if offsets[i] <= target:
                nxt = i
            else:
                break
        start = nxt

    return chunks


def chunk_document(doc: Document, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Chunk one document, tagging each chunk with its source."""
    out = []
    for i, (text, start, end) in enumerate(split_into_chunks(doc.text, size, overlap)):
        out.append(
            Chunk(
                doc_id=doc.id,
                text=text,
                index=i,
                start_char=start,
                end_char=end,
                metadata={"title": doc.title, "source": doc.source},
            )
        )
    return out
