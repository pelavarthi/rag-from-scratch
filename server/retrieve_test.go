package main

import (
	"testing"
)

func testRetriever(t *testing.T, texts []string) *Retriever {
	t.Helper()
	embedder, err := NewEmbedder(256)
	if err != nil {
		t.Fatal(err)
	}
	entries := make([]Entry, len(texts))
	for i, text := range texts {
		entries[i] = Entry{
			Chunk:  testChunk(text, "d1", nil),
			Vector: embedder.Embed(text),
		}
	}
	store, err := LoadStore(writeTestIndex(t, 256, entries))
	if err != nil {
		t.Fatal(err)
	}
	retriever, err := NewRetriever(store, embedder, 0.7, 3)
	if err != nil {
		t.Fatal(err)
	}
	return retriever
}

func TestMMRLambdaOneMatchesTopK(t *testing.T) {
	retriever := testRetriever(t, []string{
		"cats sleep all day in the sun",
		"cats sleep all day in the warm sun",
		"dogs run in the park every morning",
	})
	query := "cats sleep all day"
	plain := retriever.Store.SearchWithVectors(retriever.Embedder.Embed(query), 2, nil)
	reranked, err := retriever.Retrieve(query, 2, nil, float64Ptr(1.0), 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range plain {
		if reranked[i].Chunk.Text != plain[i].Chunk.Text {
			t.Errorf("lambda 1 hit %d = %q, want %q", i, reranked[i].Chunk.Text, plain[i].Chunk.Text)
		}
	}
}

func TestMMRDiversityPrefersDifferentSecondHit(t *testing.T) {
	retriever := testRetriever(t, []string{
		"cats sleep all day in the sun",
		"cats sleep all day in the warm sun",
		"dogs run in the park every morning",
	})
	pure, err := retriever.Retrieve("cats sleep all day", 2, nil, float64Ptr(1.0), 0)
	if err != nil {
		t.Fatal(err)
	}
	if pure[0].Chunk.Text != "cats sleep all day in the sun" || pure[1].Chunk.Text != "cats sleep all day in the warm sun" {
		t.Fatalf("pure relevance = %q, %q", pure[0].Chunk.Text, pure[1].Chunk.Text)
	}
	diverse, err := retriever.Retrieve("cats sleep all day", 2, nil, float64Ptr(0.0), 0)
	if err != nil {
		t.Fatal(err)
	}
	if diverse[0].Chunk.Text != pure[0].Chunk.Text {
		t.Errorf("diverse first hit = %q, want %q", diverse[0].Chunk.Text, pure[0].Chunk.Text)
	}
	if diverse[1].Chunk.Text != "dogs run in the park every morning" {
		t.Errorf("diverse second hit = %q, want the dogs chunk", diverse[1].Chunk.Text)
	}
}

func TestRetrieveRejectsBadKAndLambda(t *testing.T) {
	retriever := testRetriever(t, []string{"cats sleep"})
	if _, err := retriever.Retrieve("cats", 0, nil, nil, 0); err == nil {
		t.Error("k 0 should fail")
	}
	if _, err := retriever.Retrieve("cats", 1, nil, float64Ptr(1.5), 0); err == nil {
		t.Error("lambda 1.5 should fail")
	}
}

func TestNewRetrieverRejectsMismatchedDim(t *testing.T) {
	store, err := LoadStore(writeTestIndex(t, 3, []Entry{
		{Chunk: testChunk("cats", "d1", nil), Vector: []float64{1, 0, 0}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	embedder, _ := NewEmbedder(64)
	if _, err := NewRetriever(store, embedder, 0.7, 3); err == nil {
		t.Error("mismatched embedder dim should fail")
	}
}

func float64Ptr(v float64) *float64 { return &v }
