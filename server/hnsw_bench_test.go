package main

import (
	"math/rand"
	"sync"
	"testing"
)

// benchData is one shared 2000-vector index (dim 256) so every
// benchmark measures search, not setup.
var benchData struct {
	once  sync.Once
	index *HNSWIndex
	store *Store
	query []float64
}

func benchSetup() {
	benchData.once.Do(func() {
		rng := rand.New(rand.NewSource(123))
		entries := make([]Entry, 0, 2000)
		for i := 0; i < 2000; i++ {
			entries = append(entries, Entry{
				Chunk:  testChunk("chunk", "doc", nil),
				Vector: randomUnitVector(rng, 256),
			})
		}
		store := &Store{Dim: 256, entries: entries}
		benchData.store = store
		benchData.index = BuildHNSW(store, defaultHNSWOptions())
		benchData.query = randomUnitVector(rng, 256)
	})
}

// BenchmarkBruteForceSearch scans all 2000 vectors per query.
func BenchmarkBruteForceSearch(b *testing.B) {
	benchSetup()
	for i := 0; i < b.N; i++ {
		benchData.store.SearchWithVectors(benchData.query, 10, nil)
	}
}

// BenchmarkHNSWSearchEF16 searches the graph with a narrow beam.
func BenchmarkHNSWSearchEF16(b *testing.B) {
	benchSetup()
	for i := 0; i < b.N; i++ {
		benchData.index.Search(benchData.query, 10, 16)
	}
}

// BenchmarkHNSWSearchEF64 searches the graph with a wide beam.
func BenchmarkHNSWSearchEF64(b *testing.B) {
	benchSetup()
	for i := 0; i < b.N; i++ {
		benchData.index.Search(benchData.query, 10, 64)
	}
}

// BenchmarkHNSWBuild rebuilds the whole index from scratch.
func BenchmarkHNSWBuild(b *testing.B) {
	benchSetup()
	for i := 0; i < b.N; i++ {
		BuildHNSW(benchData.store, defaultHNSWOptions())
	}
}
