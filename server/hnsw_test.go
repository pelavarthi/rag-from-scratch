package main

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// randomUnitVector makes a deterministic pseudo-random unit vector: a
// random direction in dim space, normalized.
func randomUnitVector(rng *rand.Rand, dim int) []float64 {
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

func buildTestIndex(t *testing.T, n, dim int) (*HNSWIndex, *Store, [][]float64) {
	t.Helper()
	rng := rand.New(rand.NewSource(42))
	entries := make([]Entry, 0, n)
	vectors := make([][]float64, 0, n)
	for i := 0; i < n; i++ {
		vector := randomUnitVector(rng, dim)
		vectors = append(vectors, vector)
		entries = append(entries, Entry{
			Chunk:  testChunk("chunk", "doc", nil),
			Vector: vector,
		})
	}
	// Give each chunk a distinct index so tests can tell hits apart.
	for i := range entries {
		entries[i].Chunk.Index = i
	}
	store := &Store{Dim: dim, entries: entries}
	return BuildHNSW(store, defaultHNSWOptions()), store, vectors
}

func TestHNSWFindsNearest(t *testing.T) {
	// Three unit vectors in dim 3: query [1,0,0] must match the first.
	entries := []Entry{
		{Chunk: testChunk("cats sleep", "cats", nil), Vector: []float64{1, 0, 0}},
		{Chunk: testChunk("dogs run", "dogs", nil), Vector: []float64{0, 1, 0}},
		{Chunk: testChunk("birds fly", "birds", nil), Vector: []float64{0, 0, 1}},
	}
	store := &Store{Dim: 3, entries: entries}
	index := BuildHNSW(store, defaultHNSWOptions())

	results := index.Search([]float64{1, 0, 0}, 1, 16)
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0].Chunk.Text != "cats sleep" || results[0].Score != 1 {
		t.Errorf("first hit = %q score %v, want cats sleep score 1", results[0].Chunk.Text, results[0].Score)
	}
}

func TestHNSWEmptyAndSmall(t *testing.T) {
	empty := NewHNSWIndex(4, defaultHNSWOptions())
	if got := empty.Search([]float64{1, 0, 0, 0}, 5, 16); got != nil {
		t.Errorf("empty index returned %d hits, want none", len(got))
	}
	if got := empty.Search([]float64{1, 0, 0, 0}, 0, 16); got != nil {
		t.Errorf("k=0 returned %d hits, want none", len(got))
	}

	one := BuildHNSW(&Store{Dim: 2, entries: []Entry{
		{Chunk: testChunk("only", "d", nil), Vector: []float64{1, 0}},
	}}, defaultHNSWOptions())
	// Asking for more than the index holds returns what it has.
	if got := one.Search([]float64{1, 0}, 10, 16); len(got) != 1 {
		t.Errorf("k=10 on 1-vector index returned %d hits, want 1", len(got))
	}
}

func TestHNSWRecallVsBruteForce(t *testing.T) {
	// 500 random vectors in dim 64, 20 random queries: HNSW should
	// find at least 90% of the true top 10, and the top hit should
	// almost always be the true nearest neighbor.
	index, store, _ := buildTestIndex(t, 500, 64)
	rng := rand.New(rand.NewSource(7))

	totalRecall := 0.0
	top1 := 0
	queries := 20
	for q := 0; q < queries; q++ {
		query := randomUnitVector(rng, 64)
		exact := store.SearchWithVectors(query, 10, nil)
		approx := index.Search(query, 10, 64)

		exactSet := map[int]bool{}
		for _, hit := range exact {
			exactSet[hit.Chunk.Index] = true
		}
		found := 0
		for _, hit := range approx {
			if exactSet[hit.Chunk.Index] {
				found++
			}
		}
		totalRecall += float64(found) / 10
		if len(approx) > 0 && len(exact) > 0 && approx[0].Chunk.Index == exact[0].Chunk.Index {
			top1++
		}
	}
	recall := totalRecall / float64(queries)
	t.Logf("recall@10 = %.3f, top-1 match %d/%d", recall, top1, queries)
	if recall < 0.9 {
		t.Errorf("recall@10 = %.3f, want at least 0.9", recall)
	}
	if top1 < queries-2 {
		t.Errorf("top-1 matched %d/%d queries, want at least %d", top1, queries, queries-2)
	}
}

func TestHNSWRecallGrowsWithEF(t *testing.T) {
	// A wider beam should find more of the true top 10 than a narrow
	// one, on the same index and queries.
	index, store, _ := buildTestIndex(t, 500, 64)
	rng := rand.New(rand.NewSource(9))

	recall := func(ef int) float64 {
		total := 0.0
		for q := 0; q < 10; q++ {
			query := randomUnitVector(rng, 64)
			exact := store.SearchWithVectors(query, 10, nil)
			approx := index.Search(query, 10, ef)
			exactSet := map[int]bool{}
			for _, hit := range exact {
				exactSet[hit.Chunk.Index] = true
			}
			found := 0
			for _, hit := range approx {
				if exactSet[hit.Chunk.Index] {
					found++
				}
			}
			total += float64(found) / 10
		}
		return total / 10
	}
	// Fresh queries per call would differ, so reuse the same rng seed
	// by resetting it between the two runs.
	rng = rand.New(rand.NewSource(9))
	narrow := recall(10)
	rng = rand.New(rand.NewSource(9))
	wide := recall(128)
	t.Logf("recall@10 ef=10: %.3f, ef=128: %.3f", narrow, wide)
	if wide < narrow {
		t.Errorf("wider ef should not lose recall: ef=10 %.3f, ef=128 %.3f", narrow, wide)
	}
}

func TestHNSWSearchIsFasterThanBruteForce(t *testing.T) {
	// On 2000 vectors, HNSW should beat a full scan by a clear margin.
	// Times are logged, not just asserted, so a slow machine shows its
	// numbers instead of failing on noise.
	index, store, _ := buildTestIndex(t, 2000, 256)
	rng := rand.New(rand.NewSource(11))
	queries := make([][]float64, 20)
	for i := range queries {
		queries[i] = randomUnitVector(rng, 256)
	}

	timeSearch := func(search func([]float64)) float64 {
		start := time.Now()
		for _, q := range queries {
			search(q)
		}
		return float64(time.Since(start).Microseconds()) / 1000 / float64(len(queries))
	}
	brute := timeSearch(func(q []float64) { store.SearchWithVectors(q, 10, nil) })
	approx := timeSearch(func(q []float64) { index.Search(q, 10, 64) })
	t.Logf("avg search: brute force %.2f ms, HNSW %.2f ms (%.1fx faster)",
		brute, approx, brute/approx)
	if approx >= brute {
		t.Errorf("HNSW %.2f ms/query was not faster than brute force %.2f ms/query", approx, brute)
	}
}
