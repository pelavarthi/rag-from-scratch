"""Tests for document loading and directory ingestion."""

from pathlib import Path

from rag.ingest import ingest_directory, load_document


def test_load_plain_text(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Hello world.\nSecond line.\n", encoding="utf-8")
    doc = load_document(f)
    assert doc.id == "notes.txt"
    assert doc.title == "notes"
    assert "Hello world." in doc.text


def test_load_markdown_uses_first_heading_as_title(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("# My Guide\n\nSome **bold** text.\n", encoding="utf-8")
    doc = load_document(f)
    assert doc.title == "My Guide"
    assert "#" not in doc.text
    assert "Some bold text." in doc.text


def test_markdown_without_heading_falls_back_to_filename(tmp_path):
    f = tmp_path / "plain.md"
    f.write_text("Just text, no heading.\n", encoding="utf-8")
    assert load_document(f).title == "plain"


def test_ingest_directory_walks_subfolders(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Doc A.", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# Doc B\n\nBody.", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    docs = ingest_directory(tmp_path)
    assert [d.id for d in docs] == ["a.txt", "sub/b.md"]
    assert docs[1].title == "Doc B"


def test_ingest_empty_directory(tmp_path):
    assert ingest_directory(tmp_path) == []


def test_document_id_is_relative_to_root(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "b.txt"
    f.write_text("Body.", encoding="utf-8")
    assert load_document(f, root=tmp_path).id == "sub/b.txt"
