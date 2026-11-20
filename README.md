# rag-from-scratch

Retrieval-augmented generation in Python + Go: chunking, embeddings, vector search, reranking, cited answers, and an eval harness. Built from scratch, standard library first, so every step is visible.

## How it works

A RAG system answers from your documents instead of from memory. At
build time, documents are cut into chunks, each chunk is embedded into
a vector, and the vectors are saved in an index. At query time, the
question is embedded the same way, the nearest chunks are retrieved,
and those chunks — not the model's memory — become the context for the
answer, with every claim cited to a passage that was actually shown.

```
                        build time                       query time
  docs ──► ingest ──► chunk ──► embed ──► index.json
                                                      │
  question ──► embed ──► retrieve (vector + BM25, MMR)
                                                      │
                                            pack + prompt ──► answer
                                                      │
                                              verify citations ──► sources
```

Two halves, one contract: the Python pipeline builds the index, and
the Go server serves it. They are written in different languages but
agree on exactly how text becomes a vector (the same MD5 word-hash
embedder), so a query embedded in Go lands in the same space as chunk
vectors built in Python. The Python side is where the ideas live
(chunking, retrieval, answers, eval); the Go side is where the
requests land (HTTP API, HNSW index, caching, rate limiting, API keys).

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

- `server/` — a stdlib-only Go server (from `server/`: `go run . --index index.json`) that loads an index built by the Python pipeline and answers queries over HTTP. `POST /query` takes `{"query": "...", "k": 5}` and returns the k most useful chunks with scores; optional `lambda_mult` trades relevance against diversity and `where` filters by `doc_id` or metadata before ranking. `GET /health` and `GET /stats` report what the index holds, and `POST /reload` re-reads the index file without restarting.
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

## Day 12: caching, rate limiting, API keys

- `server/middleware.go` — three guards in front of the retriever, all stdlib-only. `QueryCache` remembers recent responses keyed on the whole request (query, k, MMR settings, filters), so a repeat answers in microseconds; entries expire after `--cache-ttl` (default 5m), the oldest go when the cache is full (`--cache-size`, default 1000), and `/reload` clears it so a new index never serves stale answers. `RateLimiter` is a token bucket per caller (API key, or IP when no key is sent): each caller starts with `--rate-burst` tokens and earns `--rate-limit` more per second, so short bursts pass and a sustained flood gets 429 with a `Retry-After` header. `APIKeys` stores only SHA-256 hashes of the configured keys, so a config dump never leaks a live key; with no keys configured the server stays open as before, and with keys on, `/query` and `/reload` require a valid `X-API-Key` header while `/health` and `/stats` stay public.
- `server/server.go` — `/query` checks the cache first (responses carry `cached` and an `X-Cache: HIT/MISS` header) and stores misses; `/stats` reports `cache_entries`, `cache_hits`, and `cache_misses`; `/query` and `/reload` run through the auth and rate-limit wrappers.
- `server/main.go` — `--api-keys` (or `RAG_API_KEYS`) turns auth on, `--gen-key` prints a fresh random key, `--cache-ttl`, `--cache-size`, `--rate-limit`, and `--rate-burst` tune the other two.
- Verified live: with one key configured, a request without it got 401, the first keyed request got 200 with `X-Cache: MISS`, the repeat got 200 with `X-Cache: HIT`, and a third inside the burst window got 429. `go test ./...` in `server/` passes, and the Python suite is still 149 passed.

## Day 13: benchmarks + tuning — latency, recall, index size

- `server/tune.go` — the tuning sweep, run with `go run . bench` from `server/` (saves `benchmarks/hnsw-tuning.md`). Each M is built once and searched at every ef, the way the server runs it, and recall@10 is scored against the exact brute-force top 10 on the same 50 queries. `Recommend` picks the fastest setting that clears 0.95 recall, falling back to the highest recall when none does. `SummarizeStore` / `SummarizeHNSW` size the index: vectors dominate, and each graph link costs one 8-byte id.
- `scripts/benchmark.py` — the Python end-to-end benchmark: `python scripts/benchmark.py data/sample` times chunking, embedding, and BM25 indexing separately, then times every golden question and reports query p50/p95 next to mean recall@k from the eval harness, plus the saved index size on disk (`--out` saves the report as markdown).
- Measured on 2000 random unit vectors, dim 256, 50 queries, k=10: brute force p50 1.16 ms / p99 1.89 ms. M=16 ef=64 gives p50 0.67 ms, p99 1.22 ms, recall 0.930; M=16 ef=128 reaches 0.982 at p99 1.61 ms. The sweep recommends M=32 ef=64 (recall 0.992, p99 1.56 ms, 4.9 MB total vs 3.9 MB of vectors). A narrow beam on a sparse graph is the bad corner: M=8 ef=16 finds only 0.382 of the top 10. Two honest caveats: random vectors are near-orthogonal, so these recall numbers are a pessimistic bound — real text embeddings cluster and should do better at the same ef — and at 2000 vectors HNSW barely beats brute force on p99; the graph pays off as the index grows. Defaults stay M=16, ef=64 until measured on real data.
- Measured on `data/sample` (6 chunks, dim 256): chunking 0.9 ms, embedding 0.9 ms, BM25 index 0.3 ms, query p50 0.7 ms / p95 0.9 ms, mean recall@5 1.000, index 15.1 KB. The sample is tiny, so these numbers are a smoke test that the pipeline is fast, not a load test.
- `go test ./...` in `server/` passes, and the Python suite is now 153 passed.

## Day 14: docs, demo, final polish

- `scripts/demo.py` — the scripted end-to-end demo: `python scripts/demo.py` builds the index from `data/sample`, asks all six golden questions, and prints each answer with its sources. Then it checks the three things the project is built on: retrieval finds the gold chunks (recall@5 is 1.0), every citation points at a passage that was actually in the prompt (citation precision is 1.0), and every answer is grounded in the retrieved passages (faithfulness is 1.0). A failed check exits non-zero, so the demo is also a smoke test for the whole pipeline. `--question "..."` asks a single question instead of running the golden set. `tests/test_demo.py` covers the demo itself, including a check that a gold chunk that exists nowhere in the index fails recall.
- This README is now a guide, not a build log: how it works and an architecture diagram up top, a demo section, the server API reference, how to swap in a real embedder or LLM, the measured results, and the known limits. The day-by-day sections below are kept as the record of how each piece was built.
- Final polish: the Python suite is 157 passed, `go test ./...` in `server/` passes, and `gofmt` and `go vet` are clean. One fix along the way: `scripts/demo.py` adds its own folder to `sys.path`, so the stub generator in `scripts/answer.py` resolves when the demo is loaded from the test suite. A live smoke test also caught a wrong command that had survived since Day 5: `go run ./server --index server/index.json` fails from the repo root because the Go module lives in `server/`, so the README and the `main.go` comment now say `cd server` then `go run . --index index.json`, and the tuning sweep is `go run . bench`.

## Quickstart

```bash
python -m pytest
python scripts/demo.py
python -m rag.pipeline data/sample
python scripts/answer.py data/sample "what is rag?"
python scripts/evaluate.py data/sample
python scripts/benchmark.py data/sample
python scripts/build_index.py data/sample --out server/index.json
cd server
go run . --index index.json
curl -X POST localhost:8080/query -d '{"query": "what is rag"}'
```

```bash
cd server
go test ./...
go test -bench . -run '^$'
go run . bench
```

## Demo

`python scripts/demo.py` runs the whole pipeline over the sample docs
and checks itself. Each question prints its answer, its sources, and
three scores; the run ends with `6/6 checks passed`, or exits non-zero
if any check failed. `python scripts/demo.py --question "what is rag?"`
asks one question instead.

## API reference

The Go server (from `server/`: `go run . --index index.json`) listens on
`:8080` by default.

| Endpoint | What it does |
| --- | --- |
| `GET /health` | `{"status": "ok"}` when the server is up. Public. |
| `GET /stats` | Chunk count, vector dimension, active index (brute force or HNSW), cache hits and misses. Public. |
| `POST /query` | `{"query": "...", "k": 5}` returns the k best chunks with scores, plus `took_ms` and `cached`. Optional `lambda_mult` (MMR trade-off), `fetch_k` (how many candidates to rerank), and `where` (metadata filter, e.g. `{"doc_id": "rag-intro.md"}`). Requires an API key when keys are configured; rate-limited when a limit is set. |
| `POST /reload` | Re-reads the index file (and rebuilds the HNSW graph) without restarting. Clears the query cache. |

Flags: `--index` (default `index.json`), `--addr` (default `:8080`),
`--lambda` (default `0.7`), `--hnsw` (use the approximate index),
`--ef` (default `64`), `--api-keys` (comma-separated keys, or set
`RAG_API_KEYS`; empty means open), `--cache-ttl` (default `5m`),
`--cache-size` (default `1000`), `--rate-limit` (requests per second
per caller, `0` means no limit), `--rate-burst` (default `10`),
`--gen-key` (print a fresh key and exit).

## Swapping in a real embedder / LLM

Two seams, both one function each. For embeddings, implement the
`Embedder` protocol in `rag/embeddings.py` (`dim`, `embed`,
`embed_many`) and pass it to `build_chunks` / `HybridRetriever` —
then rebuild the index, since the stored vectors have to come from the
same encoder as the queries. The Go server's `server/embed.go` must
match it the same way the hashing embedder does now. For answers, pass
any prompt-to-string callable to `AnswerGenerator` — for example a
function that calls an LLM API with the packed prompt and returns its
text. Citation verification in `rag/answers.py` already runs on the
model's output, so a real model gets the same check the stub does.

## Results

On `data/sample` (6 chunks, 6 golden questions): mean recall@5 1.000,
MRR 0.867, mean faithfulness 1.000, mean citation precision 1.000.
Answer overlap is 0.217 — the stub quotes the top passage instead of
writing a real answer, so it covers few of the reference answer's
words; that number should rise with a real model, while faithfulness
and citation precision are the ones that must not fall.

On 2000 random unit vectors, dim 256, 50 queries, k=10: brute force
p50 1.16 ms / p99 1.89 ms. M=32 ef=64 reaches 0.992 recall@10 and
M=16 ef=128 reaches 0.982, both at p99 around 1.5–1.8 ms — which of
the two is fastest flips between runs on timing noise, so re-measure
before trusting a recommendation. Full numbers are in
`server/benchmarks/hnsw-tuning.md` (re-run with `go run . bench` from
`server/`).

## Known limits

- The hashing embedder is a baseline, not a model: it only sees shared
  words, so paraphrases and synonyms do not match. Swap in a real
  encoder before judging retrieval quality on real text.
- The answer metrics are word-overlap heuristics: they catch invented
  facts and thin-air citations, but they cannot tell a right answer
  from a wrong one phrased in the same words.
- The benchmark vectors are random and near-orthogonal, so the HNSW
  recall numbers are a pessimistic bound; real embeddings cluster and
  should do better at the same `ef`. At 2000 vectors the graph barely
  beats brute force on p99 — it pays off as the index grows.
- Filtered queries (`where`) always brute-force scan, since the HNSW
  graph cannot skip vectors a filter would reject.
- The server keeps the whole index in memory and has no persistence of
  its own; the Python pipeline owns the index file.

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
