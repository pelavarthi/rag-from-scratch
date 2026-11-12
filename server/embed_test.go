package main

import (
	"math"
	"testing"
)

// TestEmbedMatchesPython checks the Go embedder against vectors computed
// by the Python HashingEmbedder, so queries embedded here land in the
// same space as chunk vectors built by the Python pipeline.
func TestEmbedMatchesPython(t *testing.T) {
	embedder, err := NewEmbedder(256)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		text string
		want map[int]float64
	}{
		{
			"cats sleep all day in the sun",
			map[int]float64{
				3: 0.377964, 32: 0.377964, 62: 0.377964, 109: 0.377964,
				176: 0.377964, 230: 0.377964, 233: 0.377964,
			},
		},
		{
			"the quick brown fox",
			map[int]float64{106: 0.5, 109: 0.5, 240: 0.5, 250: 0.5},
		},
	}
	for _, tc := range cases {
		got := embedder.Embed(tc.text)
		if len(got) != 256 {
			t.Fatalf("Embed(%q) len = %d, want 256", tc.text, len(got))
		}
		for i, value := range got {
			want, ok := tc.want[i]
			if !ok {
				if value != 0 {
					t.Errorf("Embed(%q) bucket %d = %v, want 0", tc.text, i, value)
				}
				continue
			}
			if math.Abs(value-want) > 1e-6 {
				t.Errorf("Embed(%q) bucket %d = %v, want %v", tc.text, i, value, want)
			}
		}
	}
}

func TestEmbedEmptyTextIsZeroVector(t *testing.T) {
	embedder, _ := NewEmbedder(256)
	got := embedder.Embed("")
	for i, value := range got {
		if value != 0 {
			t.Errorf("bucket %d = %v, want 0", i, value)
		}
	}
}

func TestNewEmbedderRejectsBadDim(t *testing.T) {
	if _, err := NewEmbedder(0); err == nil {
		t.Error("NewEmbedder(0) should fail")
	}
}

func TestCosine(t *testing.T) {
	if got := cosine([]float64{1, 0}, []float64{1, 0}); got != 1 {
		t.Errorf("cosine of identical vectors = %v, want 1", got)
	}
	if got := cosine([]float64{1, 0}, []float64{0, 1}); got != 0 {
		t.Errorf("cosine of unrelated vectors = %v, want 0", got)
	}
}
