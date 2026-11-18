// Command ragserver serves a RAG index over HTTP.
//
// Build an index first with the Python pipeline:
//
//	python scripts/build_index.py data/sample --out server/index.json
//
// Then run the server and ask it a question:
//
//	go run ./server --index server/index.json
//	curl -X POST localhost:8080/query -d '{"query": "what is rag"}'
//
// By default the server is open. Pass --api-keys (or set RAG_API_KEYS
// to a comma-separated list) to require an X-API-Key header on /query
// and /reload; --gen-key prints a fresh key. --rate-limit caps
// requests per second per caller, and repeated queries are served
// from an in-memory cache for --cache-ttl.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

func main() {
	indexPath := flag.String("index", "index.json", "path to the vector index JSON built by scripts/build_index.py")
	addr := flag.String("addr", ":8080", "address to listen on")
	lambda := flag.Float64("lambda", 0.7, "MMR trade-off between relevance and diversity")
	useHNSW := flag.Bool("hnsw", false, "search with the approximate HNSW index instead of brute force")
	ef := flag.Int("ef", 64, "HNSW beam width: wider is more accurate and slower")
	apiKeys := flag.String("api-keys", "", "comma-separated API keys required on /query and /reload (or set RAG_API_KEYS); empty means open")
	cacheTTL := flag.Duration("cache-ttl", 5*time.Minute, "how long query results stay cached")
	cacheSize := flag.Int("cache-size", 1000, "max cached query responses")
	rateLimit := flag.Float64("rate-limit", 0, "max requests per second per caller, 0 means no limit")
	rateBurst := flag.Int("rate-burst", 10, "burst size for the rate limiter")
	genKey := flag.Bool("gen-key", false, "print a fresh API key and exit")
	flag.Parse()

	if *genKey {
		key, err := NewAPIKey()
		if err != nil {
			log.Fatalf("generating key: %v", err)
		}
		fmt.Println(key)
		return
	}

	store, err := LoadStore(*indexPath)
	if err != nil {
		log.Fatalf("loading index: %v", err)
	}
	embedder, err := NewEmbedder(store.Dim)
	if err != nil {
		log.Fatalf("embedder: %v", err)
	}
	retriever, err := NewRetriever(store, embedder, *lambda, 3)
	if err != nil {
		log.Fatalf("retriever: %v", err)
	}
	if *useHNSW {
		retriever.HNSW = BuildHNSW(store, defaultHNSWOptions())
		retriever.HNSWEF = *ef
		log.Printf("hnsw index built: %d vectors, max level %d, ef %d",
			retriever.HNSW.Len(), retriever.HNSW.MaxLevel(), *ef)
	}
	server := NewServer(retriever, *indexPath, *useHNSW, *ef)
	server.cache = NewQueryCache(*cacheTTL, *cacheSize)
	if *rateLimit > 0 {
		server.limiter = NewRateLimiter(*rateLimit, *rateBurst)
	}
	keys := *apiKeys
	if keys == "" {
		keys = os.Getenv("RAG_API_KEYS")
	}
	if keys != "" {
		server.keys = NewAPIKeys(strings.Split(keys, ","))
		log.Printf("api keys enabled: %d key(s) configured", len(strings.Split(keys, ",")))
	}
	log.Printf("rag server listening on %s (%d chunks, dim %d)", *addr, store.Len(), store.Dim)
	log.Fatal(http.ListenAndServe(*addr, server.routes()))
}
