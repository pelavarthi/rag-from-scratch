// Benchmarks and tuning for the search index: measure latency,
// recall, and size for each HNSW setting, then recommend one.
//
// The question every index has to answer is which settings to run
// with. M controls how connected the graph is, ef controls how wide
// the search beam is, and both trade speed for recall. Guessing them
// is easy to get wrong, so this file measures instead: build one
// index per setting, time real searches, and score recall against
// the exact brute-force answer. The recommendation is the cheapest
// setting that still clears the recall bar.
package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// percentile returns the p-th percentile of sorted durations using
// the nearest-rank method: p=50 is the median, p=99 the near-worst.
// `sorted` must already be in ascending order.
func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	rank := int(math.Ceil(p / 100 * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

// StoreStats sizes the brute-force store: the vectors dominate, the
// chunk text is small next to them.
type StoreStats struct {
	Chunks      int
	Dim         int
	VectorBytes int64
	TextBytes   int64
}

func (s StoreStats) TotalBytes() int64 { return s.VectorBytes + s.TextBytes }

// SummarizeStore counts chunks and sizes the vectors and text.
func SummarizeStore(store *Store) StoreStats {
	stats := StoreStats{Chunks: store.Len(), Dim: store.Dim}
	for _, entry := range store.entries {
		stats.VectorBytes += int64(len(entry.Vector)) * 8
		stats.TextBytes += int64(len(entry.Chunk.Text))
	}
	return stats
}

// HNSWStats sizes the graph on top of the vectors: each stored link
// is one 8-byte id, so edges are the whole extra cost.
type HNSWStats struct {
	Nodes       int
	Edges       int
	VectorBytes int64
	GraphBytes  int64
}

func (s HNSWStats) TotalBytes() int64 { return s.VectorBytes + s.GraphBytes }

// SummarizeHNSW counts nodes and links in the graph.
func SummarizeHNSW(index *HNSWIndex) HNSWStats {
	stats := HNSWStats{Nodes: len(index.nodes)}
	for _, node := range index.nodes {
		stats.VectorBytes += int64(len(node.vector)) * 8
		for _, layer := range node.neighbors {
			stats.Edges += len(layer)
		}
	}
	stats.GraphBytes = int64(stats.Edges) * 8
	return stats
}

// TuningRow is one setting measured: how long it took to build, how
// fast it searches, how much of the exact answer it finds, and how
// big the graph is. Label is "brute force" for the exact baseline
// and "M=16 ef=64" for a graph setting.
type TuningRow struct {
	Label  string
	M      int
	Ef     int
	Build  time.Duration
	P50    time.Duration
	P99    time.Duration
	Recall float64
	Edges  int
	Bytes  int64
}

// TuneHNSW measures every M and ef combination on the same store and
// queries, plus a brute-force baseline row first. Recall@k is the
// share of the exact top-k chunks each setting finds, so 1.0 means
// it agrees with brute force completely. Queries are fixed vectors
// the caller supplies, so a rerun measures the same work.
//
// Each M is built once with the default construction beam and then
// searched at every ef: that matches how the server runs, where the
// graph is built once at startup and only the search beam varies per
// query (--ef flag). EfConstruction is a build-time knob, not a
// search-time one, so it is not swept here.
func TuneHNSW(store *Store, ms, efs []int, queries [][]float64, k int) []TuningRow {
	rows := []TuningRow{measureBruteForce(store, queries, k)}

	// The exact top-k per query, computed once and reused as the
	// ground truth for every setting below.
	exact := make([]map[int]bool, len(queries))
	for i, query := range queries {
		exact[i] = make(map[int]bool, k)
		for _, hit := range store.SearchWithVectors(query, k, nil) {
			exact[i][hit.Chunk.Index] = true
		}
	}

	for _, m := range ms {
		start := time.Now()
		index := BuildHNSW(store, HNSWOptions{M: m, EfConstruction: defaultHNSWOptions().EfConstruction})
		build := time.Since(start)
		stats := SummarizeHNSW(index)

		for _, ef := range efs {
			latencies := make([]time.Duration, 0, len(queries))
			found := 0
			for i, query := range queries {
				start = time.Now()
				hits := index.Search(query, k, ef)
				latencies = append(latencies, time.Since(start))
				for _, hit := range hits {
					if exact[i][hit.Chunk.Index] {
						found++
					}
				}
			}
			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
			rows = append(rows, TuningRow{
				Label:  fmt.Sprintf("M=%d ef=%d", m, ef),
				M:      m,
				Ef:     ef,
				Build:  build,
				P50:    percentile(latencies, 50),
				P99:    percentile(latencies, 99),
				Recall: float64(found) / float64(len(queries)*k),
				Edges:  stats.Edges,
				Bytes:  stats.TotalBytes(),
			})
		}
	}
	return rows
}

// measureBruteForce times the exact scan so the graph rows have
// something to beat. Its recall is 1.0 by definition.
func measureBruteForce(store *Store, queries [][]float64, k int) TuningRow {
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		start := time.Now()
		store.SearchWithVectors(query, k, nil)
		latencies = append(latencies, time.Since(start))
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	stats := SummarizeStore(store)
	return TuningRow{
		Label:  "brute force",
		P50:    percentile(latencies, 50),
		P99:    percentile(latencies, 99),
		Recall: 1,
		Bytes:  stats.TotalBytes(),
	}
}

// Recommend picks the fastest setting whose recall clears minRecall.
// When none clears it, it picks the highest recall instead — accuracy
// first, speed second. The brute-force baseline never wins: it is the
// reference, not a setting to run with.
func Recommend(rows []TuningRow, minRecall float64) TuningRow {
	var best TuningRow
	found := false
	for _, row := range rows {
		if row.M == 0 || row.Recall < minRecall {
			continue
		}
		if !found || row.P99 < best.P99 {
			best, found = row, true
		}
	}
	if found {
		return best
	}
	for _, row := range rows {
		if row.M == 0 {
			continue
		}
		if !found || row.Recall > best.Recall ||
			(row.Recall == best.Recall && row.P99 < best.P99) {
			best, found = row, true
		}
	}
	return best
}

// FormatTuningReport renders the sweep as a plain-text table, with
// the recommended setting called out at the bottom.
func FormatTuningReport(rows []TuningRow, recommended TuningRow, k int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "HNSW tuning: %d settings, recall@%d vs brute force\n", len(rows)-1, k)
	fmt.Fprintf(&b, "%-14s %10s %10s %10s %8s %10s %10s\n",
		"setting", "build", "p50", "p99", "recall", "edges", "size")
	for _, row := range rows {
		mark := " "
		if row.Label == recommended.Label {
			mark = "*"
		}
		fmt.Fprintf(&b, "%s%-13s %10s %10s %10s %8.3f %10d %10s\n",
			mark, row.Label, row.Build, row.P50, row.P99,
			row.Recall, row.Edges, formatBytes(row.Bytes))
	}
	fmt.Fprintf(&b, "recommended: %s (recall %.3f, p99 %s)\n",
		recommended.Label, recommended.Recall, recommended.P99)
	return b.String()
}

// formatBytes prints a byte count in human units.
func formatBytes(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
