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

## Day 6: hybrid search + reranking

- `rag/hybrid.py` — `BM25Index` scores chunks by exact words: term frequency times inverse document frequency, damped by chunk length, so a rare word like "quark" matters more than a word that appears everywhere. `reciprocal_rank_fusion` merges the vector ranking and the BM25 ranking using only ranks (`1 / (60 + rank)` per list), which is what makes the merge fair — a BM25 score and a cosine similarity live on different scales, so comparing raw scores would be meaningless. `HybridRetriever` runs both searches, fuses them, and reranks the fused list with MMR, so the final results are relevant, diverse, and reachable by either kind of match.
- `rag/vector_store.py` — new `vector_for` lookup, so the reranker can find the vector of a chunk the keyword search surfaced.

## Day 7: answer generation — prompt builder, context packing, citations

- `rag/answers.py` — `pack_context` fits the best chunks into a character budget, best first, truncating only the first chunk if it alone overflows so an answer never has zero context. `build_prompt` numbers each passage [1], [2], ... and instructs the model to answer only from those passages, citing every claim; with no results it says so instead of inventing passages. `parse_citations` and `verify_citations` read the model's [n] references back and drop any that were not in the prompt, so a hallucinated [99] never reaches the user. `AnswerGenerator` wires it together and takes the model as any prompt-to-string callable, so a real LLM client can be swapped in later with no other change. `Answer` carries the text plus only the verified citations; `format_sources` renders them as a readable source list.
- `scripts/answer.py` — end-to-end demo: `python scripts/answer.py data/sample "what is rag?"` retrieves with the hybrid retriever and answers with a stub that quotes the top passage.

## Day 8: evaluation harness — golden Q&A, recall@k, faithfulness

- `rag/eval.py` — scores the pipeline against a golden set instead of by feel. Retrieval: `recall_at_k`, `precision_at_k`, and reciprocal rank per question, averaged into a `RetrievalReport` (mean recall@k, mean precision@k, MRR). Answers: `faithfulness` checks each sentence's words against the retrieved passages, `citation_precision` checks the [n] references against the passages that were actually in the prompt, and `answer_overlap` checks how much of the reference answer the generated answer covers. All three are word-overlap heuristics, standard library only — they catch invented facts and thin-air citations, but they measure coverage of words, not correctness of meaning. `GoldenQuestion` names the chunks that should come back for a question, and `EvalSet` loads/saves the golden set as JSON.
- `data/sample/eval.json` — six golden questions over the sample docs, each with a reference answer and the chunk indices that answer it.
- `scripts/evaluate.py` — CLI: `python scripts/evaluate.py data/sample` builds the index, runs both reports, and prints them. `--ks 1,3,5` sets the k values, `--skip-answers` scores retrieval only.

## Day 9: chunking v2 — recursive and semantic chunking, dedup

- `rag/chunking_v2.py` — three ways to cut a document that Day 1's sentence packer cannot do. `split_recursive` splits on structure first: paragraph breaks, then line breaks, then sentences, then words, so a chunk keeps a paragraph together when it fits and only crosses a strong boundary when it has to; overlap restarts at the first piece starting inside the overlap window, so overlap 0 repeats nothing. `split_semantic` groups sentences by topic: each sentence is embedded with the pluggable `Embedder`, and a chunk keeps growing while each new sentence is similar enough (cosine similarity above a threshold) to the one before it, so a chunk ends when the topic changes instead of when a counter runs out, with a max size as a backstop. `deduplicate_chunks` drops exact copies by a hash of the normalized text, and — when an embedder is given — near-copies whose vectors are nearly identical, keeping the first of each. `chunk_document_v2` wraps either strategy and tags chunks with `"strategy"` in their metadata.
- `rag/pipeline.py` — `build_chunks` now takes `--strategy sentence|recursive|semantic`, `--dedup`, and `--threshold`, so the whole pipeline can run on the new chunkers: `python -m rag.pipeline data/sample --strategy recursive --dedup`.

## Day 10: approximate index — HNSW in Go + benchmarks

- `server/hnsw.go` — a from-scratch HNSW index (Malkov & Yashunin, 2018): vectors live on a stack of layers, sparse on top and full at the bottom. A search walks greedily down the upper layers, then runs a beam search (width `ef`) on layer 0, so it visits roughly log(n) vectors instead of all n. Distance is `1 - cosine similarity`. Levels fall off geometrically (`1/ln(M)`), so most nodes sit on layer 0 and only a few reach the top. New nodes link to their M closest neighbors per layer, pruning over-full lists back to the closest.
- The server uses it with `go run . --hnsw --ef 64` (from `server/`): unfiltered queries search the graph, filtered ones still brute-force scan, since the graph cannot skip vectors a filter would reject. `/stats` reports which index is active, and `/reload` rebuilds the graph with the new index.
- Benchmarks on 2000 random unit vectors, dim 256 (`go test -bench .` in `server/`): brute force 1.13 ms/query, HNSW ef=16 0.33 ms (3.4x faster), ef=64 0.70 ms (1.6x). Recall@10 against brute force is 1.000 at ef=64 and 0.920 at ef=10 — a wider beam buys accuracy for speed, and top-1 matches 20/20 queries. Building the graph costs ~1.8 s for 2000 vectors, paid once at startup.

## Day 11: incremental indexing — watch, add, update, delete

- `rag/incremental.py` — a manifest of one SHA-256 hash per document remembers what the index already holds, so an update only touches what moved. `scan_directory` hashes every file under the source root; `diff_manifest` sorts each doc into added, changed, deleted, or unchanged by comparing hashes (mtime is recorded but never trusted — touching a file without editing it is a no-op). `IncrementalIndexer.update` removes the old chunks of changed and deleted docs, chunks and embeds only the added and changed ones, and rewrites the index and manifest only when something moved. `watch` polls the directory and calls back only on rounds that changed anything. `open_indexer` loads an existing index and manifest, or starts fresh — a first run over a directory with no index indexes everything, exactly like a full build.
- `rag/vector_store.py` — `remove_doc` and `remove_where` drop chunks from the store, so an update can forget a document's old chunks before its new ones go in; a search never returns a chunk from a version that no longer exists.
- `scripts/build_index.py --incremental` updates an index in place instead of rebuilding it, and `scripts/watch_index.py` keeps an index fresh in the background (`--once` applies a single update and exits). The Go server picks the new index up with `POST /reload`.
- Verified on `data/sample`: an incremental update after editing one file, adding one, and deleting one embedded only 2 chunks (vs 6 for a full build), and the resulting index holds exactly the same chunks and vectors as a fresh full rebuild.

## Quickstart

```bash
python -m pytest
python -m rag.pipeline data/sample
python scripts/answer.py data/sample "what is rag?"
python scripts/evaluate.py data/sample
python scripts/build_index.py data/sample --out server/index.json
go run ./server --index server/index.json
curl -X POST localhost:8080/query -d '{"query": "what is rag"}'
```

```bash
cd server
go test ./...
go test -bench . -run '^$'
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
