"""Score the RAG pipeline against a golden set of questions.

Until now the pipeline could only be eyeballed: run a query, read the
chunks, decide by feel whether they were the right ones. That does not
scale, and it cannot tell you whether a change (a new chunker, a new
reranker) made things better or worse. This module adds a repeatable
score, in two halves:

- retrieval: for each golden question, which chunks *should* come back
  is written down in advance. recall@k is the share of those chunks
  found in the top k, precision@k is the share of the top k that was
  relevant, and MRR (mean reciprocal rank) rewards putting the first
  right chunk near the top.
- answers: is the generated answer grounded in what was retrieved?
  faithfulness checks each sentence's words against the retrieved
  passages, citation precision checks the [n] references against the
  passages that were actually in the prompt, and answer overlap checks
  how much of the reference answer the generated answer covers.

All three answer checks are word-overlap heuristics, standard library
only — they catch answers that invent facts or cite thin air, but they
cannot judge wording or nuance the way a human or a judge model could.
That is a fair trade for a harness that runs in milliseconds with no
API key, and it is honest about what it measures: coverage of words,
not correctness of meaning.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .answers import Answer, build_prompt, verify_citations
from .document import Chunk
from .vector_store import SearchResult

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[\d+\]")

# What the model says when the passages do not contain the answer. An
# abstention makes no claim, so it is skipped by the faithfulness check
# instead of scored against the context.
_ABSTENTIONS = frozenset({"i do not know", "i don't know"})

# Words that carry no meaning on their own. Dropping them keeps the
# overlap checks from rewarding filler like "the" and "is".
STOPWORDS = frozenset(
    """
    a an the and or but if then so of to in on at for with by from as is are
    was were be been being it its this that these those there here what which
    who whom how when where why do does did can could will would should may
    might must have has had not no yes i you he she we they them his her
    their our your my me him us all every each
    """.split()
)


@dataclass(frozen=True)
class ChunkRef:
    """One chunk a golden question expects: document plus position in it."""

    doc_id: str
    chunk_index: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.chunk_index)


@dataclass
class GoldenQuestion:
    """One test case: a question, its reference answer, and the chunks
    that answer it.

    `relevant` names the chunks that contain the answer, by document id
    and chunk index as produced by the pipeline. A question can have
    more than one relevant chunk when the answer spans passages.
    """

    question: str
    answer: str
    relevant: list[ChunkRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "relevant": [
                {"doc_id": ref.doc_id, "chunk_index": ref.chunk_index}
                for ref in self.relevant
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenQuestion":
        return cls(
            question=data["question"],
            answer=data["answer"],
            relevant=[
                ChunkRef(doc_id=item["doc_id"], chunk_index=item["chunk_index"])
                for item in data.get("relevant", [])
            ],
        )


@dataclass
class EvalSet:
    """A golden set of questions, loaded from or saved to one JSON file."""

    questions: list[GoldenQuestion] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    def to_dict(self) -> dict:
        return {"questions": [q.to_dict() for q in self.questions]}

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            questions=[GoldenQuestion.from_dict(item) for item in data["questions"]]
        )


@runtime_checkable
class RetrieverLike(Protocol):
    """Anything that answers a question with ranked chunks."""

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        ...


def _retrieved_keys(results: list[SearchResult]) -> list[tuple[str, int]]:
    return [(r.chunk.doc_id, r.chunk.index) for r in results]


def recall_at_k(
    results: list[SearchResult], relevant: list[ChunkRef], k: int
) -> float:
    """Share of the relevant chunks found in the top `k` results.

    With no relevant chunks listed, there is nothing to find, so the
    score is 1.0 — the question cannot fail a test it does not set.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 1.0
    wanted = {ref.key for ref in relevant}
    found = {key for key in _retrieved_keys(results[:k]) if key in wanted}
    return len(found) / len(wanted)


def precision_at_k(
    results: list[SearchResult], relevant: list[ChunkRef], k: int
) -> float:
    """Share of the top `k` results that are relevant.

    With no results at all, nothing was relevant, so the score is 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = _retrieved_keys(results[:k])
    if not top:
        return 0.0
    wanted = {ref.key for ref in relevant}
    return sum(1 for key in top if key in wanted) / len(top)


def reciprocal_rank(
    results: list[SearchResult], relevant: list[ChunkRef]
) -> float:
    """1 over the rank of the first relevant result, or 0 if none.

    A relevant chunk in 1st place scores 1, in 2nd place 0.5, and so
    on — it rewards putting the best answer first, not just somewhere
    in the list.
    """
    wanted = {ref.key for ref in relevant}
    for rank, key in enumerate(_retrieved_keys(results), start=1):
        if key in wanted:
            return 1.0 / rank
    return 0.0


def content_words(text: str) -> set[str]:
    """The meaningful words in `text`: lowercased, stopwords dropped."""
    return {word for word in _WORD.findall(text.lower()) if word not in STOPWORDS}


def split_claims(text: str) -> list[str]:
    """Split an answer into its sentences, each one a claim to check."""
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]


def faithfulness(answer_text: str, context: list[Chunk]) -> float:
    """How much of the answer's wording appears in the retrieved context.

    Each sentence is one claim. A claim scores the share of its content
    words found anywhere in the context passages, and the answer scores
    the average over its claims. Citation markers like [1] are stripped
    first — they are references, not claims. Sentences with no content
    words, and abstentions like "I do not know.", carry no claim, so
    they are skipped; an answer that makes no claims at all scores 1.0,
    since it invented nothing.
    """
    context_words: set[str] = set()
    for chunk in context:
        context_words |= content_words(chunk.text)

    claims = split_claims(_CITATION.sub(" ", answer_text))
    scores: list[float] = []
    for claim in claims:
        if claim.strip().lower().rstrip(".!?") in _ABSTENTIONS:
            continue
        words = content_words(claim)
        if not words:
            continue
        scores.append(len(words & context_words) / len(words))
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def citation_precision(answer: Answer, prompt_citations) -> float | None:
    """Share of the answer's [n] citations that were in the prompt.

    Returns None when the answer cites nothing: there is no citation to
    judge, so the question is excluded from the average rather than
    scored as a pass or a fail.
    """
    valid, invalid = verify_citations(answer.text, prompt_citations)
    if not valid and not invalid:
        return None
    return len(valid) / (len(valid) + len(invalid))


def answer_overlap(generated: str, expected: str) -> float:
    """Share of the reference answer's content words in the generated one.

    A rough check that the answer says the right things, not just
    grounded things: a faithful answer can still miss the point, and
    this catches that. With an empty reference answer there is nothing
    to cover, so the score is 1.0.
    """
    wanted = content_words(expected)
    if not wanted:
        return 1.0
    return len(wanted & content_words(generated)) / len(wanted)


@dataclass
class RetrievalRow:
    """One question's retrieval scores."""

    question: str
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0


@dataclass
class RetrievalReport:
    """Retrieval scores for a whole eval set, plus their means."""

    rows: list[RetrievalRow] = field(default_factory=list)
    ks: tuple[int, ...] = (1, 3, 5)

    @property
    def mean_recall_at_k(self) -> dict[int, float]:
        return _mean_over_rows(self.rows, lambda row: row.recall_at_k)

    @property
    def mean_precision_at_k(self) -> dict[int, float]:
        return _mean_over_rows(self.rows, lambda row: row.precision_at_k)

    @property
    def mean_reciprocal_rank(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.reciprocal_rank for row in self.rows) / len(self.rows)

    def format(self) -> str:
        """Render the report as a plain-text table with a summary."""
        lines = [f"Retrieval eval: {len(self.rows)} questions"]
        header = "question".ljust(44) + "".join(
            f"r@{k:<5}" for k in self.ks
        ) + " mrr"
        lines.append(header)
        for row in self.rows:
            short = row.question[:42]
            cells = "".join(f"{row.recall_at_k[k]:<7.2f}" for k in self.ks)
            lines.append(
                f"{short.ljust(44)}{cells}{row.reciprocal_rank:.2f}"
            )
        for k in self.ks:
            lines.append(
                f"mean recall@{k}: {self.mean_recall_at_k[k]:.3f}   "
                f"mean precision@{k}: {self.mean_precision_at_k[k]:.3f}"
            )
        lines.append(f"mean reciprocal rank: {self.mean_reciprocal_rank:.3f}")
        return "\n".join(lines)


def _mean_over_rows(
    rows: list[RetrievalRow], pick: Callable[[RetrievalRow], dict[int, float]]
) -> dict[int, float]:
    """Average a per-k score dict over rows; empty set means all zeros."""
    if not rows:
        return {}
    keys = pick(rows[0]).keys()
    return {
        k: sum(pick(row)[k] for row in rows) / len(rows) for k in keys
    }


def evaluate_retrieval(
    retriever: RetrieverLike,
    eval_set: EvalSet,
    ks: tuple[int, ...] = (1, 3, 5),
    k: int = 5,
) -> RetrievalReport:
    """Score `retriever` on every golden question.

    Each question is retrieved once at depth `k` (the largest k in
    `ks`, or `k` if larger), then recall and precision are read off the
    same list for every k — one search per question, not one per k.
    """
    if not ks:
        raise ValueError("ks must not be empty")
    fetch_k = max(max(ks), k)
    rows: list[RetrievalRow] = []
    for question in eval_set.questions:
        results = retriever.retrieve(question.question, k=fetch_k)
        rows.append(
            RetrievalRow(
                question=question.question,
                recall_at_k={
                    depth: recall_at_k(results, question.relevant, depth)
                    for depth in ks
                },
                precision_at_k={
                    depth: precision_at_k(results, question.relevant, depth)
                    for depth in ks
                },
                reciprocal_rank=reciprocal_rank(results, question.relevant),
            )
        )
    return RetrievalReport(rows=rows, ks=ks)


@dataclass
class AnswerRow:
    """One question's answer scores."""

    question: str
    faithfulness: float = 0.0
    citation_precision: float | None = None
    answer_overlap: float = 0.0


@dataclass
class AnswerReport:
    """Answer scores for a whole eval set, plus their means."""

    rows: list[AnswerRow] = field(default_factory=list)

    @property
    def mean_faithfulness(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.faithfulness for row in self.rows) / len(self.rows)

    @property
    def mean_citation_precision(self) -> float | None:
        """Mean over questions that cited something, else None."""
        cited = [r.citation_precision for r in self.rows if r.citation_precision is not None]
        if not cited:
            return None
        return sum(cited) / len(cited)

    @property
    def mean_answer_overlap(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.answer_overlap for row in self.rows) / len(self.rows)

    def format(self) -> str:
        """Render the report as a plain-text table with a summary."""
        lines = [f"Answer eval: {len(self.rows)} questions"]
        lines.append("question".ljust(44) + "faithful  cited  overlap")
        for row in self.rows:
            short = row.question[:42]
            cited = (
                f"{row.citation_precision:.2f}"
                if row.citation_precision is not None
                else "n/a"
            )
            lines.append(
                f"{short.ljust(44)}{row.faithfulness:<10.2f}"
                f"{cited:<7}{row.answer_overlap:.2f}"
            )
        lines.append(f"mean faithfulness: {self.mean_faithfulness:.3f}")
        mean_cited = self.mean_citation_precision
        lines.append(
            "mean citation precision: "
            + (f"{mean_cited:.3f}" if mean_cited is not None else "n/a")
        )
        lines.append(f"mean answer overlap: {self.mean_answer_overlap:.3f}")
        return "\n".join(lines)


def evaluate_answers(
    answer_fn: Callable[[str, list[SearchResult]], Answer],
    retriever: RetrieverLike,
    eval_set: EvalSet,
    k: int = 5,
) -> AnswerReport:
    """Score generated answers on every golden question.

    `answer_fn` takes a question and its retrieved chunks and returns
    an `Answer`. Faithfulness is checked against the chunks that were
    actually retrieved, not the golden ones — an answer can only be
    faithful to what it was shown.
    """
    rows: list[AnswerRow] = []
    for question in eval_set.questions:
        results = retriever.retrieve(question.question, k=k)
        answer = answer_fn(question.question, results)
        # Rebuild the prompt citations to check against: the answer was
        # generated from these passages, in this order.
        prompt = build_prompt(question.question, results)
        rows.append(
            AnswerRow(
                question=question.question,
                faithfulness=faithfulness(
                    answer.text, [r.chunk for r in results]
                ),
                citation_precision=citation_precision(answer, prompt.citations),
                answer_overlap=answer_overlap(answer.text, question.answer),
            )
        )
    return AnswerReport(rows=rows)
