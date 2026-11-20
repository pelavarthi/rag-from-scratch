"""Scripted end-to-end demo of the whole RAG pipeline.

Builds the index from a directory of documents, asks every golden
question, and prints each answer with its sources. Then it checks the
three things the project is built on:

- retrieval finds the gold chunks (recall@5 is 1.0 on the sample set),
- every citation in an answer points at a passage that was actually
  shown to the model (citation precision is 1.0),
- every answer is grounded in the retrieved passages (faithfulness
  is 1.0).

A check that fails exits non-zero, so the demo doubles as a smoke
test for the whole pipeline:

    python scripts/demo.py
    python scripts/demo.py data/sample --question "what is rag?"
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
# The scripts folder itself goes on the path too, so the stub in
# answer.py resolves when this module is loaded from the test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import stub_generate  # noqa: E402  (scripts/ holds the stub)
from rag.answers import (  # noqa: E402
    Answer,
    AnswerGenerator,
    build_prompt,
    format_sources,
)
from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.eval import (  # noqa: E402
    EvalSet,
    citation_precision,
    faithfulness,
    recall_at_k,
)
from rag.hybrid import HybridRetriever  # noqa: E402
from rag.pipeline import build_chunks  # noqa: E402
from rag.vector_store import SearchResult  # noqa: E402

DEFAULT_K = 5


@dataclass
class DemoCheck:
    """One golden question, its answer, and the three scores."""

    question: str
    answer: Answer
    results: list[SearchResult]
    recall: float
    citation_precision: float
    faithfulness: float

    @property
    def passed(self) -> bool:
        return (
            self.recall == 1.0
            and self.citation_precision == 1.0
            and self.faithfulness == 1.0
        )

    def format(self) -> str:
        lines = [f"Q: {self.question}", f"A: {self.answer.text}", ""]
        lines.append(format_sources(self.answer.citations))
        lines.append(
            f"checks: recall@5 {self.recall:.3f}, "
            f"citation precision {self.citation_precision:.3f}, "
            f"faithfulness {self.faithfulness:.3f} "
            f"-> {'PASS' if self.passed else 'FAIL'}"
        )
        return "\n".join(lines)


@dataclass
class DemoResult:
    """The outcome of running the demo over a golden set."""

    checks: list[DemoCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def format(self) -> str:
        parts = [check.format() for check in self.checks]
        passed = sum(1 for c in self.checks if c.passed)
        parts.append("")
        parts.append(
            f"{passed}/{len(self.checks)} checks passed"
            + (" — all checks passed." if self.passed else " — see FAIL above.")
        )
        return "\n\n".join(parts)


def build_retriever(source_dir: str) -> HybridRetriever:
    chunks = build_chunks(source_dir)
    return HybridRetriever.from_chunks(chunks, HashingEmbedder(dim=256))


def answer_question(
    retriever: HybridRetriever, question: str, k: int = DEFAULT_K
) -> tuple[Answer, list[SearchResult]]:
    results = retriever.retrieve(question, k=k)
    answer = AnswerGenerator(generate=stub_generate).answer(question, results)
    return answer, results


def check_question(
    retriever: HybridRetriever, question, k: int = DEFAULT_K
) -> DemoCheck:
    """Answer one golden question and score it on the three checks."""
    answer, results = answer_question(retriever, question.question, k=k)
    prompt = build_prompt(question.question, results)
    precision = citation_precision(answer, prompt.citations)
    return DemoCheck(
        question=question.question,
        answer=answer,
        results=results,
        recall=recall_at_k(results, question.relevant, k),
        citation_precision=0.0 if precision is None else precision,
        faithfulness=faithfulness(answer.text, [r.chunk for r in results]),
    )


def run_demo(
    source_dir: str, eval_path: str, k: int = DEFAULT_K
) -> DemoResult:
    """Run the demo over every golden question in `eval_path`."""
    eval_set = EvalSet.load(eval_path)
    retriever = build_retriever(source_dir)
    return DemoResult(
        checks=[check_question(retriever, q, k=k) for q in eval_set.questions]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo the RAG pipeline end to end, with checks."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="data/sample",
        help="directory of .txt / .md files (default: data/sample)",
    )
    parser.add_argument(
        "--eval",
        default="data/sample/eval.json",
        help="golden questions JSON (default: data/sample/eval.json)",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="chunks to retrieve"
    )
    parser.add_argument(
        "--question",
        help="ask one question instead of running the golden set",
    )
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")

    if args.question:
        retriever = build_retriever(args.source_dir)
        answer, _ = answer_question(retriever, args.question, k=args.k)
        print(answer.text)
        print()
        print(format_sources(answer.citations))
        return

    result = run_demo(args.source_dir, args.eval, k=args.k)
    print(result.format())
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
