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
package main

import (
	"flag"
	"log"
	"net/http"
)

func main() {
	indexPath := flag.String("index", "index.json", "path to the vector index JSON built by scripts/build_index.py")
	addr := flag.String("addr", ":8080", "address to listen on")
	lambda := flag.Float64("lambda", 0.7, "MMR trade-off between relevance and diversity")
	flag.Parse()

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
	server := NewServer(retriever, *indexPath)
	log.Printf("rag server listening on %s (%d chunks, dim %d)", *addr, store.Len(), store.Dim)
	log.Fatal(http.ListenAndServe(*addr, server.routes()))
}
