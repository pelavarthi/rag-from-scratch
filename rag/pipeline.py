"""Glue it together: a directory of files in, a list of chunks out."""

import argparse
import json
from pathlib import Path

from .chunking import chunk_document
from .document import Chunk
from .ingest import ingest_directory


def build_chunks(
    source_dir: str | Path, size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Ingest every document under `source_dir` and chunk each one."""
    chunks: list[Chunk] = []
    for doc in ingest_directory(source_dir):
        chunks.extend(chunk_document(doc, size=size, overlap=overlap))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a directory into chunks.")
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument("--size", type=int, default=500, help="max chunk chars")
    parser.add_argument("--overlap", type=int, default=50, help="overlap chars")
    parser.add_argument("--out", help="optional path to write chunks as JSONL")
    args = parser.parse_args()

    chunks = build_chunks(args.source_dir, size=args.size, overlap=args.overlap)
    print(f"{len(chunks)} chunks")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
