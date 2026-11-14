"""Turn retrieved chunks into a cited answer.

Retrieval finds the right passages, but a bare list of chunks is not an
answer. This module is the last step of the pipeline:

- context packing fits the best chunks into the model's budget, best
  first, so the prompt never overflows and the most relevant passage is
  never the one that gets cut;
- the prompt builder numbers each passage and tells the model to answer
  only from those passages, citing them as [1], [2], ...;
- the answer parser reads the model's [n] citations back and checks
  them against the passages that were actually in the prompt, so a
  hallucinated citation like [99] is dropped instead of shown to the user.

The model itself is pluggable: `AnswerGenerator` takes any callable
that maps a prompt string to an answer string, just as `Embedder` takes
any encoder. There is no real LLM in this repo yet — the demo script
uses a stub that quotes the top passage — so the wiring can be tested
without an API key.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .document import Chunk
from .vector_store import SearchResult

DEFAULT_MAX_CHARS = 4000

SYSTEM_PROMPT = (
    "Answer the question using only the passages below. "
    "If the passages do not contain the answer, say you do not know. "
    "Cite every claim with the passage number in square brackets, "
    "like [1] or [2]. Do not cite a passage number that is not listed."
)


@dataclass
class Citation:
    """One passage the answer may cite.

    `number` is the 1-based label used in the prompt and in the answer
    text ([1], [2], ...). `source` is a human-readable name for where
    the passage came from, usually the document title.
    """

    number: int
    doc_id: str
    chunk_index: int
    source: str
    start_char: int
    end_char: int

    @property
    def label(self) -> str:
        return f"[{self.number}]"


@dataclass
class PackedPrompt:
    """A prompt plus the passages it actually contains.

    `citations` are the passages in the prompt, in the same order and
    numbering as the prompt text, so an answer's [n] can be checked
    against them directly.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    """A generated answer with verified citations.

    `citations` holds only the passages the answer actually cited and
    that were in the prompt, in order of first mention. Anything the
    model cited that was not in the prompt is dropped, never shown.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)


def pack_context(
    results: list[SearchResult],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[SearchResult], int]:
    """Fit the best results into `max_chars`, best first.

    Returns the results that fit and how many were dropped. A chunk is
    dropped when it does not fit in what is left of the budget. The one
    exception is the first chunk: if even it is too long, it is kept
    truncated, because an answer with no context at all is worse than
    an answer with a partial passage.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    packed: list[SearchResult] = []
    used = 0
    for result in results:
        length = len(result.chunk.text)
        if used + length <= max_chars:
            packed.append(result)
            used += length
        elif not packed and length > max_chars:
            cut = result.chunk
            truncated = Chunk(
                doc_id=cut.doc_id,
                text=cut.text[: max_chars - 3] + "...",
                index=cut.index,
                start_char=cut.start_char,
                end_char=cut.end_char,
                metadata=cut.metadata,
            )
            packed.append(SearchResult(chunk=truncated, score=result.score))
            used = max_chars
        # else: does not fit, drop it and keep trying shorter later ones
    return packed, len(results) - len(packed)


def build_prompt(
    question: str,
    results: list[SearchResult],
    max_chars: int = DEFAULT_MAX_CHARS,
    system: str = SYSTEM_PROMPT,
) -> PackedPrompt:
    """Build the prompt for `question` from the retrieved `results`.

    Each passage is numbered from 1 in the order given (best first),
    and the question comes after the passages so the model reads the
    evidence before the ask. With no results, the prompt says so
    plainly instead of inventing passages.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    packed, _dropped = pack_context(results, max_chars=max_chars)

    citations = [
        Citation(
            number=i + 1,
            doc_id=result.chunk.doc_id,
            chunk_index=result.chunk.index,
            source=result.chunk.metadata.get("title", result.chunk.doc_id),
            start_char=result.chunk.start_char,
            end_char=result.chunk.end_char,
        )
        for i, result in enumerate(packed)
    ]

    lines = [system, "", "Passages:"]
    if packed:
        for citation, result in zip(citations, packed):
            lines.append(f"{citation.label} {result.chunk.text}")
    else:
        lines.append("(no passages were retrieved)")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return PackedPrompt(text="\n".join(lines), citations=citations)


_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(text: str) -> list[int]:
    """The passage numbers cited in `text`, in order, without repeats.

    Only the numbers are returned; whether they point at a real
    passage is a separate check, so this is also useful for counting
    citations before verifying them.
    """
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def verify_citations(
    text: str, citations: list[Citation]
) -> tuple[list[Citation], list[int]]:
    """Split the citations in `text` into real ones and bad ones.

    Returns the cited passages (in order of first mention) and the
    numbers that do not match any passage in `citations`, such as [0]
    or a [99] the model invented.
    """
    by_number = {citation.number: citation for citation in citations}
    valid: list[Citation] = []
    invalid: list[int] = []
    for number in parse_citations(text):
        if number in by_number:
            valid.append(by_number[number])
        else:
            invalid.append(number)
    return valid, invalid


def format_sources(citations: list[Citation]) -> str:
    """Render the cited passages as a source list for the user.

    Each line names the document and the chunk, so a reader can find
    the passage the answer came from without re-running the search.
    """
    if not citations:
        return "Sources: none"
    lines = ["Sources:"]
    for citation in citations:
        lines.append(
            f"{citation.label} {citation.source} "
            f"(chunk {citation.chunk_index})"
        )
    return "\n".join(lines)


@dataclass
class AnswerGenerator:
    """Generate a cited answer: pack, prompt, call the model, verify.

    `generate` is any function from a prompt string to an answer
    string — a real LLM client, or a stub in tests. `max_chars` is the
    context budget handed to the prompt builder.
    """

    generate: Callable[[str], str]
    max_chars: int = DEFAULT_MAX_CHARS

    def answer(
        self, question: str, results: list[SearchResult]
    ) -> Answer:
        """Answer `question` from `results`, with verified citations."""
        prompt = build_prompt(question, results, max_chars=self.max_chars)
        text = self.generate(prompt.text)
        valid, _invalid = verify_citations(text, prompt.citations)
        return Answer(text=text, citations=valid)
