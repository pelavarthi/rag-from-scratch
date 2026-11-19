package main

import (
	"math/rand"
	"strings"
	"testing"
	"time"
)

func TestPercentile(t *testing.T) {
	// Nearest-rank: p50 of 10 values is the 5th, p99 the 10th.
	durations := make([]time.Duration, 10)
	for i := range durations {
		durations[i] = time.Duration(i+1) * time.Millisecond
	}
	if got := percentile(durations, 50); got != 5*time.Millisecond {
		t.Errorf("p50 = %v, want 5ms", got)
	}
	if got := percentile(durations, 99); got != 10*time.Millisecond {
		t.Errorf("p99 = %v, want 10ms", got)
	}
	if got := percentile(durations, 0); got != time.Millisecond {
		t.Errorf("p0 = %v, want 1ms", got)
	}
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("empty percentile = %v, want 0", got)
	}
}

func TestSummarizeStore(t *testing.T) {
	entries := []Entry{
		{Chunk: testChunk("hello", "a", nil), Vector: []float64{1, 0}},
		{Chunk: testChunk("world!", "b", nil), Vector: []float64{0, 1}},
	}
	stats := SummarizeStore(&Store{Dim: 2, entries: entries})
	if stats.Chunks != 2 || stats.Dim != 2 {
		t.Errorf("stats = %+v, want 2 chunks dim 2", stats)
	}
	// 2 vectors x 2 dims x 8 bytes, plus 5 + 6 bytes of text.
	if stats.VectorBytes != 32 || stats.TextBytes != 11 {
		t.Errorf("bytes = %d vector + %d text, want 32 + 11",
			stats.VectorBytes, stats.TextBytes)
	}
}

func TestSummarizeHNSWCountsEdges(t *testing.T) {
	_, store, _ := buildTestIndex(t, 20, 8)
	index := BuildHNSW(store, defaultHNSWOptions())
	stats := SummarizeHNSW(index)
	if stats.Nodes != 20 {
		t.Errorf("nodes = %d, want 20", stats.Nodes)
	}
	if stats.Edges == 0 {
		t.Error("edges = 0, want a connected graph")
	}
	if stats.TotalBytes() <= stats.VectorBytes {
		t.Error("graph bytes should add to the vector bytes")
	}
}

func TestTuneHNSW(t *testing.T) {
	_, store, _ := buildTestIndex(t, 200, 32)
	rng := rand.New(rand.NewSource(7))
	queries := make([][]float64, 10)
	for i := range queries {
		queries[i] = randomUnitVector(rng, 32)
	}

	rows := TuneHNSW(store, []int{8, 16}, []int{16, 64}, queries, 5)
	// One brute-force baseline plus 2 x 2 settings.
	if len(rows) != 5 {
		t.Fatalf("got %d rows, want 5", len(rows))
	}
	if rows[0].Label != "brute force" || rows[0].Recall != 1 {
		t.Errorf("baseline = %+v, want brute force with recall 1", rows[0])
	}
	for _, row := range rows[1:] {
		if row.Recall < 0 || row.Recall > 1 {
			t.Errorf("%s recall = %f, want in [0, 1]", row.Label, row.Recall)
		}
		if row.P99 < row.P50 {
			t.Errorf("%s p99 %v < p50 %v", row.Label, row.P99, row.P50)
		}
		if row.Build <= 0 || row.Edges == 0 || row.Bytes == 0 {
			t.Errorf("%s looks unmeasured: %+v", row.Label, row)
		}
	}

	recommended := Recommend(rows, 0.95)
	if recommended.M == 0 {
		t.Error("recommendation picked the brute-force baseline")
	}
	if recommended.Recall < 0.95 {
		// Only acceptable when no setting cleared the bar: then the
		// recommendation must be the highest-recall setting.
		for _, row := range rows[1:] {
			if row.Recall > recommended.Recall {
				t.Errorf("recommended recall %.3f below %s %.3f",
					recommended.Recall, row.Label, row.Recall)
			}
		}
	}
}

func TestRecommendFallsBackToHighestRecall(t *testing.T) {
	rows := []TuningRow{
		{Label: "brute force", Recall: 1},
		{Label: "M=8 ef=16", M: 8, Ef: 16, Recall: 0.6, P99: time.Millisecond},
		{Label: "M=16 ef=64", M: 16, Ef: 64, Recall: 0.9, P99: 2 * time.Millisecond},
	}
	got := Recommend(rows, 0.95)
	if got.Label != "M=16 ef=64" {
		t.Errorf("recommended %s, want the highest-recall setting", got.Label)
	}
}

func TestFormatTuningReport(t *testing.T) {
	rows := []TuningRow{
		{Label: "brute force", P50: time.Millisecond, P99: 2 * time.Millisecond, Recall: 1, Bytes: 1024},
		{Label: "M=16 ef=64", M: 16, Ef: 64, Build: time.Second, P50: 100 * time.Microsecond, P99: 200 * time.Microsecond, Recall: 0.98, Edges: 500, Bytes: 2048},
	}
	report := FormatTuningReport(rows, rows[1], 10)
	if !strings.Contains(report, "M=16 ef=64") || !strings.Contains(report, "recommended") {
		t.Errorf("report missing setting or recommendation:\n%s", report)
	}
	if !strings.Contains(report, "2.0 KB") {
		t.Errorf("report should show the size in KB:\n%s", report)
	}
}
