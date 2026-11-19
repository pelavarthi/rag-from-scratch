// Command `go run ./server bench` runs the HNSW tuning sweep on a
// synthetic index (2000 random unit vectors, dim 256), prints the
// latency / recall / size table, and saves it to
// benchmarks/hnsw-tuning.md. It measures the search path only —
// embedding and HTTP overhead are not in these numbers.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// syntheticVector is a deterministic pseudo-random unit vector, the
// same shape as the test helper but available to the bench command.
func syntheticVector(rng *rand.Rand, dim int) []float64 {
	vector := make([]float64, dim)
	length := 0.0
	for i := range vector {
		vector[i] = rng.NormFloat64()
		length += vector[i] * vector[i]
	}
	length = math.Sqrt(length)
	for i := range vector {
		vector[i] /= length
	}
	return vector
}

func runBench() error {
	const n, dim, numQueries, k = 2000, 256, 50, 10
	rng := rand.New(rand.NewSource(123))

	entries := make([]Entry, 0, n)
	for i := 0; i < n; i++ {
		entries = append(entries, Entry{
			Chunk:  Chunk{DocID: "synthetic", Index: i, Text: "synthetic"},
			Vector: syntheticVector(rng, dim),
		})
	}
	store := &Store{Dim: dim, entries: entries}

	queries := make([][]float64, 0, numQueries)
	for i := 0; i < numQueries; i++ {
		queries = append(queries, syntheticVector(rng, dim))
	}

	start := time.Now()
	rows := TuneHNSW(store, []int{8, 16, 32}, []int{16, 32, 64, 128}, queries, k)
	recommended := Recommend(rows, 0.95)
	elapsed := time.Since(start)

	stats := SummarizeStore(store)
	var b strings.Builder
	fmt.Fprintf(&b, "# HNSW tuning report\n\n")
	fmt.Fprintf(&b, "Synthetic index: %d vectors, dim %d, %s of vectors. ",
		stats.Chunks, stats.Dim, formatBytes(stats.VectorBytes))
	fmt.Fprintf(&b, "Sweep of M in {8, 16, 32} x search ef in {16, 32, 64, 128}; "+
		"each M is built once and searched at every ef, as the server does. "+
		"%d queries, k=%d, run in %s.\n\n", numQueries, k, elapsed)
	fmt.Fprintf(&b, "```\n%s```\n\n", FormatTuningReport(rows, recommended, k))
	fmt.Fprintf(&b, "Note: random vectors are near-orthogonal, so these recall "+
		"numbers are a pessimistic bound — real text embeddings cluster, "+
		"and recall on those should be higher at the same ef. At this "+
		"small scale HNSW barely beats brute force; the graph pays off "+
		"as the index grows. Defaults stay M=16, ef=64 until measured "+
		"on real data.\n")
	report := b.String()
	fmt.Print(report)

	if err := os.MkdirAll("benchmarks", 0o755); err != nil {
		return fmt.Errorf("creating benchmarks dir: %w", err)
	}
	path := filepath.Join("benchmarks", "hnsw-tuning.md")
	if err := os.WriteFile(path, []byte(report), 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	fmt.Printf("saved %s\n", path)
	return nil
}
