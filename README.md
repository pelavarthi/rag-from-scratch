# rag-from-scratch

Retrieval-augmented generation in Python + Go: chunking, embeddings, vector search, reranking, cited answers, and an eval harness. Built from scratch, standard library first, so every step is visible.

## Day 1: scaffold + ingestion and chunking

- `rag/document.py` — `Document` and `Chunk` types. A chunk carries its source document id plus character offsets, so it can always be traced back.
- `rag/ingest.py` — loads `.txt` and `.md` files (Markdown headings become titles, formatting is stripped). `ingest_directory` walks a folder recursively.
- `rag/chunking.py` — sentence-aware chunker: packs whole sentences up to a size limit, with an overlap so facts straddling a boundary appear in both chunks.
- `rag/pipeline.py` — `build_chunks(dir)` ties it together; also a CLI: `python -m rag.pipeline data/sample --out chunks.jsonl`.

## Day 2: embeddings — baseline encoder + pluggable interface

- `rag/embeddings.py` — `Embedder` protocol (`dim`, `embed`, `embed_many`) so any encoder can be swapped in later. `HashingEmbedder` is the stdlib-only baseline: words are lowercased, hashed with MD5 into 256 buckets, counted, and L2-normalized, so similar texts score higher on cosine similarity. Also `cosine_similarity`, `normalize`, and `embed_chunks`, which pairs each chunk with its vector.

## Day 3: vector store — in-memory index, cosine search, persistence

- `rag/vector_store.py` — `VectorStore` holds embedded chunks and searches them by cosine similarity, best first (brute-force scan, the baseline Day 10's approximate index has to beat). `from_chunks` builds a store in one call, `search_text` embeds the query and searches, and `save`/`load` persist the whole index to one JSON file so it can be reused without re-embedding. `SearchResult` pairs each hit's chunk with its score.

## Day 4: retrieval — top-k, MMR diversity, metadata filters

- `rag/retrieval.py` — `Retriever` ties the store and embedder together: it embeds the question, fetches more candidates than it needs, and returns the `k` most useful chunks. `matches_metadata` narrows a search before ranking (by `doc_id` or any chunk metadata, one value or a list of acceptable values), and `mmr_rerank` applies maximal marginal relevance so the results cover different angles instead of repeating the same fact: each new pick balances relevance to the query against how different it is from the chunks already picked, with `lambda_mult` setting the trade-off.
- `rag/vector_store.py` — `search` and `search_text` take an optional `where` predicate that filters chunks before scoring; `search_with_vectors` also returns each hit's vector, which the reranker needs to compare candidates against each other.

## Day 5: Go retrieval API server

- `server/` — a stdlib-only Go server (`go run ./server --index server/index.json`) that loads an index built by the Python pipeline and answers queries over HTTP. `POST /query` takes `{"query": "...", "k": 5}` and returns the k most useful chunks with scores; optional `lambda_mult` trades relevance against diversity and `where` filters by `doc_id` or metadata before ranking. `GET /health` and `GET /stats` report what the index holds, and `POST /reload` re-reads the index file without restarting.
- `server/embed.go` — a port of the Python `HashingEmbedder` (same MD5 word buckets, same L2 normalization), so a query embedded in Go lands in the same space as chunk vectors built in Python; a test checks it against vectors computed by the Python encoder.
- `server/store.go` — loads the index JSON, searches it by cosine similarity, and filters by metadata. `server/retrieve.go` — the `Retriever` (embed, over-fetch, MMR rerank) ported from `rag/retrieval.py`.
- `scripts/build_index.py` — builds the index JSON the server loads: `python scripts/build_index.py data/sample --out server/index.json`. `server/index.json` is a sample index built from `data/sample` so the server runs out of the box.

## Quickstart

```bash
python -m pytest
python -m rag.pipeline data/sample
python scripts/build_index.py data/sample --out server/index.json
go run ./server --index server/index.json
curl -X POST localhost:8080/query -d '{"query": "what is rag"}'
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
