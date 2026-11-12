"""Build a vector index JSON for the Go retrieval server.

Reads a directory of documents with the Python pipeline, embeds the
chunks with the hashing baseline, and writes the index the server loads:

    python scripts/build_index.py data/sample --out server/index.json
"""

import argparse
import sys
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embeddings import HashingEmbedder
from rag.pipeline import build_chunks
from rag.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a vector index JSON for the Go server."
    )
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument("--out", required=True, help="path to write the index JSON")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension")
    args = parser.parse_args()

    chunks = build_chunks(args.source_dir)
    store = VectorStore.from_chunks(chunks, HashingEmbedder(dim=args.dim))
    store.save(args.out)
    print(f"{len(store)} chunks from {args.source_dir} -> {args.out}")


if __name__ == "__main__":
    main()
