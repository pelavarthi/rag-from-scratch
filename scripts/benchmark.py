"""Time the RAG pipeline end to end and report latency, recall, and index size.

One run answers the three questions that matter before tuning anything:
how long does each stage take, how good is retrieval, and how big is
the index on disk. Stages are timed separately — ingest and chunking,
embedding, BM25 indexing — then every golden question is retrieved
once and timed, so the report shows p50/p95 query latency next to
mean recall@k from the eval harness:

    python scripts/benchmark.py data/sample
    python scripts/benchmark.py data/sample --out benchmarks/pipeline.md

Timings are wall-clock on this machine, so compare runs on the same
machine, not across machines. The index is saved to a temporary file
only to measure its size; nothing is written to the repo unless
--out is given.
"""

import argparse
import math
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.eval import EvalSet, evaluate_retrieval  # noqa: E402
from rag.hybrid import BM25Index, HybridRetriever  # noqa: E402
from rag.pipeline import build_chunks  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: p=50 is the median, p=95 the near-worst."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def format_bytes(n: int) -> str:
    """Print a byte count in human units."""
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def _timed(fn):
    """Run `fn` and return (result, elapsed seconds)."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


@dataclass
class PipelineBenchmark:
    """One benchmark run: stage timings, query latencies, recall, size."""

    chunks: int
    dim: int
    chunk_seconds: float
    embed_seconds: float
    bm25_seconds: float
    query_seconds: list[float] = field(default_factory=list)
    mean_recall_at_k: float = 0.0
    k: int = 5
    index_bytes: int = 0

    @property
    def p50_ms(self) -> float:
        return percentile([t * 1000 for t in self.query_seconds], 50)

    @property
    def p95_ms(self) -> float:
        return percentile([t * 1000 for t in self.query_seconds], 95)

    def format(self) -> str:
        """Render the run as a plain-text table."""
        lines = [
            f"Pipeline benchmark: {self.chunks} chunks, dim {self.dim}",
            f"  chunking:    {self.chunk_seconds * 1000:7.1f} ms",
            f"  embedding:   {self.embed_seconds * 1000:7.1f} ms",
            f"  bm25 index:  {self.bm25_seconds * 1000:7.1f} ms",
            f"  query p50:   {self.p50_ms:7.1f} ms",
            f"  query p95:   {self.p95_ms:7.1f} ms",
            f"  mean recall@{self.k}: {self.mean_recall_at_k:.3f}",
            f"  index size:  {format_bytes(self.index_bytes)}",
        ]
        return "\n".join(lines)


def run_benchmark(
    source_dir: str | Path,
    eval_path: str | Path,
    k: int = 5,
    dim: int = 256,
) -> PipelineBenchmark:
    """Benchmark the full pipeline on `source_dir` and `eval_path`."""
    chunks, chunk_seconds = _timed(lambda: build_chunks(source_dir))
    embedder = HashingEmbedder(dim=dim)
    store, embed_seconds = _timed(
        lambda: VectorStore.from_chunks(chunks, embedder)
    )
    bm25, bm25_seconds = _timed(lambda: BM25Index.from_chunks(chunks))
    retriever = HybridRetriever(store=store, embedder=embedder, bm25=bm25)

    eval_set = EvalSet.load(eval_path)
    query_seconds = []
    for question in eval_set.questions:
        _, elapsed = _timed(lambda q=question: retriever.retrieve(q.question, k=k))
        query_seconds.append(elapsed)

    report = evaluate_retrieval(retriever, eval_set, ks=(k,), k=k)
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index.json"
        store.save(index_path)
        index_bytes = index_path.stat().st_size

    return PipelineBenchmark(
        chunks=len(chunks),
        dim=dim,
        chunk_seconds=chunk_seconds,
        embed_seconds=embed_seconds,
        bm25_seconds=bm25_seconds,
        query_seconds=query_seconds,
        mean_recall_at_k=report.mean_recall_at_k[k],
        k=k,
        index_bytes=index_bytes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline latency, recall, and index size."
    )
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument(
        "--eval",
        default="data/sample/eval.json",
        help="golden questions JSON (default: data/sample/eval.json)",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="chunks retrieved per question"
    )
    parser.add_argument(
        "--dim", type=int, default=256, help="embedding dimension"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="also save the report as markdown to this path",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    result = run_benchmark(args.source_dir, args.eval, k=args.k, dim=args.dim)
    text = result.format()
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Pipeline benchmark\n\n```\n" + text + "\n```\n",
                        encoding="utf-8")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
