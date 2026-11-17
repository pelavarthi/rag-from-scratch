"""Build a vector index JSON for the Go retrieval server.

Reads a directory of documents with the Python pipeline, embeds the
chunks with the hashing baseline, and writes the index the server loads:

    python scripts/build_index.py data/sample --out server/index.json

With --incremental, an existing index is updated in place instead of
rebuilt: only added and changed documents are re-embedded, deleted
ones are dropped, and unchanged ones are skipped. A manifest next to
the index (``<out>.manifest.json``) remembers what was indexed.
"""

import argparse
import sys
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embeddings import HashingEmbedder
from rag.incremental import open_indexer
from rag.pipeline import build_chunks
from rag.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a vector index JSON for the Go server."
    )
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument("--out", required=True, help="path to write the index JSON")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="update an existing index in place instead of rebuilding it",
    )
    args = parser.parse_args()

    if args.incremental:
        indexer = open_indexer(
            args.source_dir, args.out, HashingEmbedder(dim=args.dim)
        )
        summary = indexer.update()
        print(f"{len(indexer.store)} chunks in {args.out} ({summary})")
        return

    chunks = build_chunks(args.source_dir)
    store = VectorStore.from_chunks(chunks, HashingEmbedder(dim=args.dim))
    store.save(args.out)
    print(f"{len(store)} chunks from {args.source_dir} -> {args.out}")


if __name__ == "__main__":
    main()
