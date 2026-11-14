"""Ask a question over a directory of documents, with citations.

End-to-end demo of the Day 7 answer step: retrieve the best chunks
with the hybrid retriever, then answer from them. There is no real LLM
in this repo, so the demo uses a stub generator that quotes the first
sentence of the top passage and cites it — enough to show the wiring,
from retrieval through prompt building to verified citations:

    python scripts/answer.py data/sample "what is rag?"
"""

import argparse
import re
import sys
from pathlib import Path

# Running a script from scripts/ puts that folder on sys.path, not the
# repo root, so add the root (the parent of this file's folder) first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.answers import AnswerGenerator, format_sources
from rag.embeddings import HashingEmbedder
from rag.hybrid import HybridRetriever
from rag.pipeline import build_chunks


def stub_generate(prompt: str) -> str:
    """Pretend to be a model: quote the top passage's first sentence.

    The real model's job is to read the numbered passages and write an
    answer citing them. This stub does the simplest version of that:
    it takes the first sentence of passage [1] and cites it.
    """
    match = re.search(r"\n\[1\] (.*?)(?:\n\[2\] |\n\nQuestion:)", prompt, re.DOTALL)
    if not match:
        return "I do not know."
    first_sentence = match.group(1).strip().split(". ")[0]
    return f"{first_sentence}. [1]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a question over a directory of documents."
    )
    parser.add_argument("source_dir", help="directory of .txt / .md files")
    parser.add_argument("question", help="question to answer")
    parser.add_argument("--k", type=int, default=3, help="chunks to retrieve")
    args = parser.parse_args()

    chunks = build_chunks(args.source_dir)
    retriever = HybridRetriever.from_chunks(chunks, HashingEmbedder(dim=256))
    results = retriever.retrieve(args.question, k=args.k)
    answer = AnswerGenerator(generate=stub_generate).answer(
        args.question, results
    )
    print(answer.text)
    print()
    print(format_sources(answer.citations))


if __name__ == "__main__":
    main()
