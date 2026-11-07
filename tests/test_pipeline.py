"""End-to-end test: directory in, chunks out."""

from rag.pipeline import build_chunks


def test_build_chunks_end_to_end(tmp_path):
    (tmp_path / "a.txt").write_text(
        "RAG stands for retrieval-augmented generation. "
        "It grounds a language model in your own documents.",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "# Vector Search\n\nDocuments become vectors. "
        "Similar questions find similar vectors.",
        encoding="utf-8",
    )
    chunks = build_chunks(tmp_path, size=60, overlap=0)
    assert len(chunks) >= 2
    assert {c.doc_id for c in chunks} == {"a.txt", "b.md"}
    assert all(c.text for c in chunks)
    assert all(c.start_char < c.end_char for c in chunks)
