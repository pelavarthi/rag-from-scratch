"""Glue it together: a directory of files in, a list of chunks out."""

import argparse
import json
from pathlib import Path

from .chunking import chunk_document
from .chunking_v2 import chunk_document_v2, deduplicate_chunks
from .document import Chunk
from .embeddings import Embedder, HashingEmbedder
from .ingest import ingest_directory


def build_chunks(
    source_dir: str | Path,
    size: int = 500,
    overlap: int = 50,
    strategy: str = "sentence",
    dedup: bool = False,
    embedder: Embedder | None = None,
    threshold: float = 0.5,
) -> list[Chunk]:
    """Ingest every document under `source_dir` and chunk each one.

    `strategy` picks the chunker: "sentence" (Day 1, length + sentences),
    "recursive" (structure first), or "semantic" (topic first, needs an
    embedder — a HashingEmbedder is used when none is given). With
    `dedup`, exact and near-duplicate chunks are dropped afterwards.
    """
    chunks: list[Chunk] = []
    for doc in ingest_directory(source_dir):
        if strategy == "sentence":
            chunks.extend(chunk_document(doc, size=size, overlap=overlap))
        elif strategy in ("recursive", "semantic"):
            if strategy == "semantic" and embedder is None:
                embedder = HashingEmbedder()
            chunks.extend(
                chunk_document_v2(
                    doc,
                    strategy=strategy,
                    size=size,
                    overlap=overlap,
                    embedder=embedder,
                    threshold=threshold,
                )
            )
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")
    if dedup:
        chunks = deduplicate_chunks(chunks, embedder=embedder)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a directory into chunks.")
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument("--size", type=int, default=500, help="max chunk chars")
    parser.add_argument("--overlap", type=int, default=50, help="overlap chars")
    parser.add_argument(
        "--strategy",
        default="sentence",
        choices=["sentence", "recursive", "semantic"],
        help="chunking strategy",
    )
    parser.add_argument("--dedup", action="store_true", help="drop duplicate chunks")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="semantic similarity threshold for topic changes",
    )
    parser.add_argument("--out", help="optional path to write chunks as JSONL")
    args = parser.parse_args()

    chunks = build_chunks(
        args.source_dir,
        size=args.size,
        overlap=args.overlap,
        strategy=args.strategy,
        dedup=args.dedup,
        threshold=args.threshold,
    )
    print(f"{len(chunks)} chunks")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
