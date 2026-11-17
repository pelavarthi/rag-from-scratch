"""Watch a directory and keep its vector index fresh.

Polls the source directory and applies incremental updates as files
are added, changed, or deleted, so the index tracks the documents
without a rebuild. The Go server picks the new index up with
``POST /reload`` — or restart it against the same index file.

    python scripts/watch_index.py data/sample --index server/index.json
    python scripts/watch_index.py data/sample --index server/index.json --once
"""

import argparse
import sys
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embeddings import HashingEmbedder
from rag.incremental import UpdateSummary, open_indexer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a directory and keep its vector index fresh."
    )
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument("--index", required=True, help="index JSON to keep updated")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between scans (default 2)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="apply one update and exit instead of watching",
    )
    args = parser.parse_args()

    indexer = open_indexer(args.source_dir, args.index, HashingEmbedder(dim=args.dim))

    def report(summary: UpdateSummary) -> None:
        print(f"{len(indexer.store)} chunks in {args.index} ({summary})", flush=True)

    if args.once:
        summary = indexer.update()
        print(f"{len(indexer.store)} chunks in {args.index} ({summary})")
        return

    print(f"watching {args.source_dir} -> {args.index} (Ctrl-C to stop)")
    first = indexer.update()
    print(f"{len(indexer.store)} chunks in {args.index} ({first})")
    indexer.watch(interval=args.interval, on_update=report)


if __name__ == "__main__":
    main()
