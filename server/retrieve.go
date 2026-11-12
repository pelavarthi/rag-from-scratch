package main

import (
	"errors"
	"fmt"
	"math"
)

// SearchResult is one hit: the chunk that matched and how close it was.
type SearchResult struct {
	Chunk Chunk
	Score float64
}

// MMR picks k results from candidates by maximal marginal relevance:
// the first pick is the most relevant chunk, and each later pick is the
// candidate that best balances relevance against novelty. lambda 1 is
// plain top-k; lambda 0 is pure diversity.
func MMR(query []float64, candidates []ScoredEntry, k int, lambda float64) ([]SearchResult, error) {
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if lambda < 0 || lambda > 1 {
		return nil, errors.New("lambda must be between 0 and 1")
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	picked := []ScoredEntry{candidates[0]}
	rest := append([]ScoredEntry(nil), candidates[1:]...)
	for len(rest) > 0 && len(picked) < k {
		best := 0
		bestScore := math.Inf(-1)
		for i, item := range rest {
			overlap := math.Inf(-1)
			for _, other := range picked {
				if sim := cosine(item.Vector, other.Vector); sim > overlap {
					overlap = sim
				}
			}
			score := lambda*item.Score - (1-lambda)*overlap
			if score > bestScore {
				bestScore, best = score, i
			}
		}
		picked = append(picked, rest[best])
		rest = append(rest[:best], rest[best+1:]...)
	}

	results := make([]SearchResult, len(picked))
	for i, entry := range picked {
		results[i] = SearchResult{Chunk: entry.Chunk, Score: entry.Score}
	}
	return results, nil
}

// Retriever searches a store for a question: embed, over-fetch, rerank.
type Retriever struct {
	Store           *Store
	Embedder        *Embedder
	Lambda          float64
	FetchMultiplier int
}

func NewRetriever(store *Store, embedder *Embedder, lambda float64, fetchMultiplier int) (*Retriever, error) {
	if embedder.Dim != store.Dim {
		return nil, fmt.Errorf(
			"embedder dim %d does not match store dim %d",
			embedder.Dim, store.Dim,
		)
	}
	if lambda < 0 || lambda > 1 {
		return nil, errors.New("lambda must be between 0 and 1")
	}
	if fetchMultiplier < 1 {
		return nil, errors.New("fetch multiplier must be at least 1")
	}
	return &Retriever{
		Store:           store,
		Embedder:        embedder,
		Lambda:          lambda,
		FetchMultiplier: fetchMultiplier,
	}, nil
}

// Retrieve returns the k most useful chunks for query, best first.
// `where` narrows the search before ranking, `lambda` overrides the
// retriever's default for this query, and `fetchK` overrides how many
// candidates MMR chooses from (raised to k when smaller).
func (r *Retriever) Retrieve(query string, k int, where map[string]any, lambda *float64, fetchK int) ([]SearchResult, error) {
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	lam := r.Lambda
	if lambda != nil {
		lam = *lambda
	}
	if lam < 0 || lam > 1 {
		return nil, errors.New("lambda must be between 0 and 1")
	}
	if fetchK <= 0 {
		fetchK = k * r.FetchMultiplier
	}
	if fetchK < k {
		fetchK = k
	}

	queryVector := r.Embedder.Embed(query)
	candidates := r.Store.SearchWithVectors(queryVector, fetchK, where)
	return MMR(queryVector, candidates, k, lam)
}
