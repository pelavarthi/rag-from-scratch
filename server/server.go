package main

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// Server wraps a Retriever with HTTP handlers. A read/write lock guards
// the retriever so /reload can swap the index while queries run.
type Server struct {
	mu        sync.RWMutex
	retriever *Retriever
	indexPath string
	useHNSW   bool
	hnswEF    int
	// cache, limiter, and keys are optional: nil means that piece is off.
	cache   *QueryCache
	limiter *RateLimiter
	keys    *APIKeys
}

func NewServer(retriever *Retriever, indexPath string, useHNSW bool, hnswEF int) *Server {
	return &Server{retriever: retriever, indexPath: indexPath, useHNSW: useHNSW, hnswEF: hnswEF}
}

// buildRetriever loads the index and, when HNSW is on, builds the graph
// over it, keeping the retriever's MMR settings.
func (s *Server) buildRetriever(store *Store, lambda float64, fetchMultiplier int) (*Retriever, error) {
	embedder, err := NewEmbedder(store.Dim)
	if err != nil {
		return nil, err
	}
	retriever, err := NewRetriever(store, embedder, lambda, fetchMultiplier)
	if err != nil {
		return nil, err
	}
	if s.useHNSW {
		retriever.HNSW = BuildHNSW(store, defaultHNSWOptions())
		retriever.HNSWEF = s.hnswEF
	}
	return retriever, nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

type queryRequest struct {
	Query      string         `json:"query"`
	K          int            `json:"k"`
	LambdaMult *float64       `json:"lambda_mult"`
	FetchK     int            `json:"fetch_k"`
	Where      map[string]any `json:"where"`
}

type queryHit struct {
	DocID     string         `json:"doc_id"`
	Text      string         `json:"text"`
	Index     int            `json:"index"`
	StartChar int            `json:"start_char"`
	EndChar   int            `json:"end_char"`
	Metadata  map[string]any `json:"metadata"`
	Score     float64        `json:"score"`
}

type queryResponse struct {
	Query   string     `json:"query"`
	Results []queryHit `json:"results"`
	TookMs  int64      `json:"took_ms"`
	Cached  bool       `json:"cached"`
}

// handleQuery answers one search: {"query": "...", "k": 5} in, the k
// most useful chunks out, best first. Optional "lambda_mult" trades
// relevance against diversity and "where" filters by doc_id or metadata.
func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req queryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.Query == "" {
		writeError(w, http.StatusBadRequest, "query must not be empty")
		return
	}
	if req.K == 0 {
		req.K = 5
	}

	start := time.Now()
	// A repeat of the same request answers from the cache, so the
	// retriever only runs on a miss. The key covers the whole request,
	// including filters and MMR settings.
	if s.cache != nil {
		if hits, ok := s.cache.Get(cacheKey(req)); ok {
			w.Header().Set("X-Cache", "HIT")
			writeJSON(w, http.StatusOK, queryResponse{
				Query:   req.Query,
				Results: hits,
				TookMs:  time.Since(start).Milliseconds(),
				Cached:  true,
			})
			return
		}
	}

	s.mu.RLock()
	results, err := s.retriever.Retrieve(req.Query, req.K, req.Where, req.LambdaMult, req.FetchK)
	s.mu.RUnlock()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	hits := make([]queryHit, 0, len(results))
	for _, result := range results {
		hits = append(hits, queryHit{
			DocID:     result.Chunk.DocID,
			Text:      result.Chunk.Text,
			Index:     result.Chunk.Index,
			StartChar: result.Chunk.StartChar,
			EndChar:   result.Chunk.EndChar,
			Metadata:  result.Chunk.Metadata,
			Score:     result.Score,
		})
	}
	if s.cache != nil {
		s.cache.Put(cacheKey(req), hits)
	}
	w.Header().Set("X-Cache", "MISS")
	writeJSON(w, http.StatusOK, queryResponse{
		Query:   req.Query,
		Results: hits,
		TookMs:  time.Since(start).Milliseconds(),
	})
}

// handleHealth reports the server is up and what index it holds.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"chunks": s.retriever.Store.Len(),
		"dim":    s.retriever.Store.Dim,
	})
}

// handleStats reports what the index holds: chunk count, dim, documents.
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	indexType := "brute-force"
	if s.retriever.HNSW != nil {
		indexType = "hnsw"
	}
	stats := map[string]any{
		"chunks": s.retriever.Store.Len(),
		"dim":    s.retriever.Store.Dim,
		"docs":   s.retriever.Store.DocIDs(),
		"index":  indexType,
	}
	if s.cache != nil {
		entries, hits, misses := s.cache.Stats()
		stats["cache_entries"] = entries
		stats["cache_hits"] = hits
		stats["cache_misses"] = misses
	}
	writeJSON(w, http.StatusOK, stats)
}

// handleReload re-reads the index file from disk and swaps it in, so a
// rebuilt index takes effect without restarting the server.
func (s *Server) handleReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	store, err := LoadStore(s.indexPath)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "reloading index: "+err.Error())
		return
	}
	s.mu.RLock()
	lambda := s.retriever.Lambda
	fetchMultiplier := s.retriever.FetchMultiplier
	s.mu.RUnlock()
	retriever, err := s.buildRetriever(store, lambda, fetchMultiplier)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.mu.Lock()
	s.retriever = retriever
	s.mu.Unlock()
	// The index changed, so cached answers may be stale.
	if s.cache != nil {
		s.cache.Clear()
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "reloaded",
		"chunks": store.Len(),
		"dim":    store.Dim,
	})
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/stats", s.handleStats)
	// /query and /reload sit behind API-key auth and rate limiting;
	// /health and /stats stay public so monitors need no key.
	mux.HandleFunc("/query", s.requireKey(s.rateLimit(s.handleQuery)))
	mux.HandleFunc("/reload", s.requireKey(s.handleReload))
	return mux
}
