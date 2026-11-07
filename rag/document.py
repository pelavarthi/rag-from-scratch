"""Core types for the RAG pipeline: documents in, chunks out."""

from dataclasses import dataclass, field


@dataclass
class Document:
    """One source file, loaded into memory.

    `id` is stable across runs: the path relative to the ingested root.
    `metadata` carries anything the loader knew (file type, title).
    """

    id: str
    title: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """One retrievable piece of a document.

    `start_char` / `end_char` are offsets into the source document's text,
    so a chunk can always be traced back to where it came from.
    """

    doc_id: str
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
