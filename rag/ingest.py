"""Load documents from disk: plain text and Markdown files.

A loader turns a file into a Document. Markdown gets light cleanup — the
first `#` heading becomes the title, and heading markers are stripped so
they don't pollute the text that later gets embedded.
"""

import re
from pathlib import Path

from .document import Document

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _title_from_markdown(text: str, fallback: str) -> str:
    match = _HEADING.search(text)
    return match.group(1).strip() if match else fallback


def _clean_markdown(text: str) -> str:
    text = _HEADING.sub(r"\1", text)
    text = re.sub(r"[*_`]{1,3}", "", text)
    return text


def load_document(path: str | Path, root: str | Path | None = None) -> Document:
    """Load one file. `root` sets the document id (path relative to root)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    doc_id = str(path.relative_to(root)) if root else path.name

    if path.suffix.lower() in {".md", ".markdown"}:
        title = _title_from_markdown(raw, path.stem)
        text = _clean_markdown(raw)
    else:
        title = path.stem
        text = raw

    return Document(
        id=doc_id,
        title=title,
        source=str(path),
        text=text.strip(),
        metadata={"extension": path.suffix.lower()},
    )


def ingest_directory(
    root: str | Path, extensions: set[str] | None = None
) -> list[Document]:
    """Load every supported file under `root`, sorted for stable order."""
    root = Path(root)
    extensions = extensions or TEXT_EXTENSIONS
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            docs.append(load_document(path, root=root))
    return docs
