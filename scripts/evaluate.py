"""Score the RAG pipeline on a golden set of questions.

Builds the index from a directory of documents, loads the golden
questions, and prints two reports: retrieval scores (recall@k,
precision@k, MRR) and answer scores (faithfulness, citation
precision, answer overlap). Answers come from the same stub generator
as scripts/answer.py — it quotes the top passage — since there is no
real LLM in this repo yet:

    python scripts/evaluate.py data/sample
    python scripts/evaluate.py data/sample --eval data/sample/eval.json --k 3
"""

import argparse
import sys
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer import stub_generate  # noqa: E402  (scripts/ holds the stub)
from rag.answers import AnswerGenerator  # noqa: E402
from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.eval import EvalSet, evaluate_answers, evaluate_retrieval  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402
from rag.pipeline import build_chunks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score retrieval and answers on a golden question set."
    )
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument(
        "--eval",
        default="data/sample/eval.json",
        help="golden questions JSON (default: data/sample/eval.json)",
    )
    parser.add_argument(
        "--ks",
        default="1,3,5",
        help="comma-separated k values for recall/precision (default: 1,3,5)",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="chunks retrieved per question"
    )
    parser.add_argument(
        "--skip-answers",
        action="store_true",
        help="only score retrieval, skip answer generation",
    )
    args = parser.parse_args()

    ks = tuple(int(part) for part in args.ks.split(",") if part.strip())
    if not ks or any(k <= 0 for k in ks):
        parser.error("--ks must be a comma-separated list of positive ints")
    if args.k <= 0:
        parser.error("--k must be positive")

    eval_set = EvalSet.load(args.eval)
    chunks = build_chunks(args.source_dir)
    retriever = HybridRetriever.from_chunks(chunks, HashingEmbedder(dim=256))

    report = evaluate_retrieval(retriever, eval_set, ks=ks, k=args.k)
    print(report.format())

    if not args.skip_answers:
        generator = AnswerGenerator(generate=stub_generate)
        answer_report = evaluate_answers(
            generator.answer, retriever, eval_set, k=args.k
        )
        print()
        print(answer_report.format())


if __name__ == "__main__":
    main()
