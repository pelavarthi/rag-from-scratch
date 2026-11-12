package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeTestIndex(t *testing.T, dim int, entries []Entry) string {
	t.Helper()
	data, err := json.Marshal(map[string]any{"dim": dim, "entries": entries})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "index.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func testChunk(text, docID string, metadata map[string]any) Chunk {
	return Chunk{
		DocID: docID, Text: text, Index: 0,
		StartChar: 0, EndChar: len(text), Metadata: metadata,
	}
}

func TestLoadStoreAndSearch(t *testing.T) {
	// Three unit vectors in dim 3: query [1,0,0] matches the first best.
	entries := []Entry{
		{Chunk: testChunk("cats sleep", "cats", nil), Vector: []float64{1, 0, 0}},
		{Chunk: testChunk("dogs run", "dogs", nil), Vector: []float64{0, 1, 0}},
		{Chunk: testChunk("birds fly", "birds", nil), Vector: []float64{0, 0, 1}},
	}
	store, err := LoadStore(writeTestIndex(t, 3, entries))
	if err != nil {
		t.Fatal(err)
	}
	if store.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", store.Len())
	}
	if store.Dim != 3 {
		t.Fatalf("Dim = %d, want 3", store.Dim)
	}

	results := store.SearchWithVectors([]float64{1, 0, 0}, 2, nil)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].Chunk.Text != "cats sleep" || results[0].Score != 1 {
		t.Errorf("first hit = %q score %v, want cats sleep score 1", results[0].Chunk.Text, results[0].Score)
	}
}

func TestLoadStoreRejectsBadDim(t *testing.T) {
	entries := []Entry{
		{Chunk: testChunk("cats sleep", "cats", nil), Vector: []float64{1, 0}},
	}
	if _, err := LoadStore(writeTestIndex(t, 3, entries)); err == nil {
		t.Error("vector dim 2 in a dim 3 index should fail")
	}
}

func TestLoadStoreMissingFile(t *testing.T) {
	if _, err := LoadStore(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("missing index file should fail")
	}
}

func TestSearchFiltersByDocIDAndMetadata(t *testing.T) {
	entries := []Entry{
		{Chunk: testChunk("cats sleep", "d1", map[string]any{"title": "Cats"}), Vector: []float64{1, 0}},
		{Chunk: testChunk("cats nap", "d2", map[string]any{"title": "Cats"}), Vector: []float64{1, 0}},
		{Chunk: testChunk("cats purr", "d3", map[string]any{"title": "Dogs"}), Vector: []float64{1, 0}},
	}
	store, err := LoadStore(writeTestIndex(t, 2, entries))
	if err != nil {
		t.Fatal(err)
	}

	byDoc := store.SearchWithVectors([]float64{1, 0}, 5, map[string]any{"doc_id": "d2"})
	if len(byDoc) != 1 || byDoc[0].Chunk.DocID != "d2" {
		t.Errorf("doc_id filter returned %d hits, want only d2", len(byDoc))
	}

	byTitle := store.SearchWithVectors([]float64{1, 0}, 5, map[string]any{"title": []any{"Dogs", "Cats"}})
	if len(byTitle) != 3 {
		t.Errorf("title list filter returned %d hits, want 3", len(byTitle))
	}

	none := store.SearchWithVectors([]float64{1, 0}, 5, map[string]any{"doc_id": "nope"})
	if len(none) != 0 {
		t.Errorf("no-match filter returned %d hits, want 0", len(none))
	}
}

func TestDocIDs(t *testing.T) {
	entries := []Entry{
		{Chunk: testChunk("a", "b.md", nil), Vector: []float64{1, 0}},
		{Chunk: testChunk("b", "a.md", nil), Vector: []float64{1, 0}},
		{Chunk: testChunk("c", "a.md", nil), Vector: []float64{1, 0}},
	}
	store, err := LoadStore(writeTestIndex(t, 2, entries))
	if err != nil {
		t.Fatal(err)
	}
	got := store.DocIDs()
	if len(got) != 2 || got[0] != "a.md" || got[1] != "b.md" {
		t.Errorf("DocIDs() = %v, want [a.md b.md]", got)
	}
}
