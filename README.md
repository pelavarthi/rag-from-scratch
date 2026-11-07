# rag-from-scratch

Retrieval-augmented generation in Python + Go: chunking, embeddings, vector search, reranking, cited answers, and an eval harness. Built from scratch, standard library first, so every step is visible.

## Day 1: scaffold + ingestion and chunking

- `rag/document.py` — `Document` and `Chunk` types. A chunk carries its source document id plus character offsets, so it can always be traced back.
- `rag/ingest.py` — loads `.txt` and `.md` files (Markdown headings become titles, formatting is stripped). `ingest_directory` walks a folder recursively.
- `rag/chunking.py` — sentence-aware chunker: packs whole sentences up to a size limit, with an overlap so facts straddling a boundary appear in both chunks.
- `rag/pipeline.py` — `build_chunks(dir)` ties it together; also a CLI: `python -m rag.pipeline data/sample --out chunks.jsonl`.

## Quickstart

```bash
python -m pytest
python -m rag.pipeline data/sample
```

## Plan

1. Scaffold + ingestion and chunking
2. Embeddings: baseline encoder + pluggable interface
3. Vector store: in-memory index, cosine search, persistence
4. Retrieval: top-k, MMR diversity, metadata filters
5. Go retrieval API server
6. Hybrid search + reranking
7. Answer generation with citations
8. Evaluation harness
9. Chunking v2: recursive and semantic chunking, dedup
10. Approximate index: HNSW in Go + benchmarks
11. Incremental indexing
12. Caching, rate limiting, API keys
13. Benchmarks + tuning
14. Docs, demo, final polish
