package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Chunk is one retrievable piece of a document, as saved by the Python
// pipeline. Metadata values are usually strings; anything else is
// compared by its printed form.
type Chunk struct {
	DocID     string         `json:"doc_id"`
	Text      string         `json:"text"`
	Index     int            `json:"index"`
	StartChar int            `json:"start_char"`
	EndChar   int            `json:"end_char"`
	Metadata  map[string]any `json:"metadata"`
}

// Entry pairs a chunk with its vector.
type Entry struct {
	Chunk  Chunk     `json:"chunk"`
	Vector []float64 `json:"vector"`
}

// ScoredEntry is a search hit plus the vector the reranker needs.
type ScoredEntry struct {
	Chunk  Chunk
	Score  float64
	Vector []float64
}

// Store is an in-memory index of embedded chunks, loaded from the JSON
// file the Python pipeline writes.
type Store struct {
	Dim     int
	entries []Entry
}

// LoadStore reads an index written by VectorStore.save.
func LoadStore(path string) (*Store, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Dim     int     `json:"dim"`
		Entries []Entry `json:"entries"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	if raw.Dim <= 0 {
		return nil, fmt.Errorf("index %s has non-positive dim %d", path, raw.Dim)
	}
	for _, entry := range raw.Entries {
		if len(entry.Vector) != raw.Dim {
			return nil, fmt.Errorf(
				"chunk in %s has vector dim %d, want %d",
				entry.Chunk.DocID, len(entry.Vector), raw.Dim,
			)
		}
	}
	return &Store{Dim: raw.Dim, entries: raw.Entries}, nil
}

// Len returns how many chunks the store holds.
func (s *Store) Len() int {
	return len(s.entries)
}

// DocIDs lists every document id in the index, sorted.
func (s *Store) DocIDs() []string {
	seen := map[string]bool{}
	for _, entry := range s.entries {
		seen[entry.Chunk.DocID] = true
	}
	ids := make([]string, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// matchesWhere reports whether a chunk satisfies every condition in
// `where`, mirroring rag.retrieval.matches_metadata: "doc_id" matches
// the chunk's document id, any other key matches its metadata. A value
// is one expected value or a list of acceptable values.
func matchesWhere(chunk Chunk, where map[string]any) bool {
	for key, expected := range where {
		var actual any
		var ok bool
		if key == "doc_id" {
			actual = chunk.DocID
			ok = true
		} else {
			actual, ok = chunk.Metadata[key]
		}
		if !ok {
			return false
		}
		if list, isList := expected.([]any); isList {
			match := false
			for _, value := range list {
				if fmt.Sprint(value) == fmt.Sprint(actual) {
					match = true
					break
				}
			}
			if !match {
				return false
			}
		} else if fmt.Sprint(expected) != fmt.Sprint(actual) {
			return false
		}
	}
	return true
}

// SearchWithVectors returns the k chunks closest to the query vector,
// best first, each with its vector for reranking. `where` narrows the
// search before ranking; nil means no filter.
func (s *Store) SearchWithVectors(query []float64, k int, where map[string]any) []ScoredEntry {
	scored := make([]ScoredEntry, 0, len(s.entries))
	for _, entry := range s.entries {
		if where != nil && !matchesWhere(entry.Chunk, where) {
			continue
		}
		scored = append(scored, ScoredEntry{
			Chunk:  entry.Chunk,
			Score:  cosine(query, entry.Vector),
			Vector: entry.Vector,
		})
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if k > len(scored) {
		k = len(scored)
	}
	return scored[:k]
}
