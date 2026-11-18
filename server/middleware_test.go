package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func postQuery(t *testing.T, server *Server, body string, key string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(body))
	req.RemoteAddr = "127.0.0.1:1234"
	if key != "" {
		req.Header.Set("X-API-Key", key)
	}
	rec := httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)
	return rec
}

func TestQueryCacheHit(t *testing.T) {
	server := testServer(t)
	server.cache = NewQueryCache(5*time.Minute, 100)

	first := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	if first.Code != http.StatusOK || first.Header().Get("X-Cache") != "MISS" {
		t.Fatalf("first query: status %d, X-Cache %q, want 200 MISS", first.Code, first.Header().Get("X-Cache"))
	}
	var firstResp queryResponse
	if err := json.NewDecoder(first.Body).Decode(&firstResp); err != nil {
		t.Fatal(err)
	}
	if firstResp.Cached {
		t.Error("first response has cached=true, want false")
	}

	second := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	if second.Code != http.StatusOK || second.Header().Get("X-Cache") != "HIT" {
		t.Fatalf("second query: status %d, X-Cache %q, want 200 HIT", second.Code, second.Header().Get("X-Cache"))
	}
	var secondResp queryResponse
	if err := json.NewDecoder(second.Body).Decode(&secondResp); err != nil {
		t.Fatal(err)
	}
	if !secondResp.Cached {
		t.Error("second response has cached=false, want true")
	}
	if len(secondResp.Results) != 1 || secondResp.Results[0].Text != firstResp.Results[0].Text {
		t.Errorf("cached results %v, want the same as the first response", secondResp.Results)
	}
}

func TestQueryCacheMissOnDifferentRequest(t *testing.T) {
	server := testServer(t)
	server.cache = NewQueryCache(5*time.Minute, 100)

	postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	rec := postQuery(t, server, `{"query": "dogs run", "k": 1}`, "")
	if rec.Header().Get("X-Cache") != "MISS" {
		t.Errorf("different query: X-Cache %q, want MISS", rec.Header().Get("X-Cache"))
	}
	// Same query but a different k is a different request too.
	rec = postQuery(t, server, `{"query": "cats sleep", "k": 2}`, "")
	if rec.Header().Get("X-Cache") != "MISS" {
		t.Errorf("different k: X-Cache %q, want MISS", rec.Header().Get("X-Cache"))
	}
}

func TestQueryCacheExpires(t *testing.T) {
	server := testServer(t)
	server.cache = NewQueryCache(10*time.Millisecond, 100)

	postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	time.Sleep(20 * time.Millisecond)
	rec := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	if rec.Header().Get("X-Cache") != "MISS" {
		t.Errorf("expired entry: X-Cache %q, want MISS", rec.Header().Get("X-Cache"))
	}
}

func TestQueryCacheEvictsOldest(t *testing.T) {
	cache := NewQueryCache(5*time.Minute, 2)
	cache.Put("a", nil)
	cache.Put("b", nil)
	cache.Put("c", nil)
	if _, ok := cache.Get("a"); ok {
		t.Error("oldest entry survived a full cache, want it evicted")
	}
	if _, ok := cache.Get("b"); !ok {
		t.Error("recent entry was evicted, want it kept")
	}
	entries, hits, misses := cache.Stats()
	if entries != 2 {
		t.Errorf("entries = %d, want 2", entries)
	}
	if hits != 1 || misses != 1 {
		t.Errorf("hits = %d, misses = %d, want 1 and 1", hits, misses)
	}
}

func TestQueryCacheClear(t *testing.T) {
	cache := NewQueryCache(5*time.Minute, 10)
	cache.Put("a", nil)
	cache.Clear()
	if _, ok := cache.Get("a"); ok {
		t.Error("entry survived Clear, want it gone")
	}
}

func TestRateLimiterBlocksBurst(t *testing.T) {
	server := testServer(t)
	server.limiter = NewRateLimiter(0.01, 2) // refills too slowly to matter here

	for i := 0; i < 2; i++ {
		rec := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
		if rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d, want 200", i+1, rec.Code)
		}
	}
	rec := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("third request: status %d, want 429", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("429 response has no Retry-After header")
	}
}

func TestRateLimiterRefills(t *testing.T) {
	limiter := NewRateLimiter(100, 1) // 100 tokens/sec, burst of 1
	if !limiter.Allow("a") {
		t.Fatal("first Allow is false, want true")
	}
	if limiter.Allow("a") {
		t.Fatal("second Allow is true right away, want false")
	}
	time.Sleep(20 * time.Millisecond)
	if !limiter.Allow("a") {
		t.Fatal("Allow after refill is false, want true")
	}
	// A different caller has its own bucket.
	if !limiter.Allow("b") {
		t.Fatal("other caller was throttled, want its own bucket")
	}
}

func TestAPIKeyAuth(t *testing.T) {
	server := testServer(t)
	server.keys = NewAPIKeys([]string{"secret-key"})

	rec := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("no key: status %d, want 401", rec.Code)
	}
	rec = postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "wrong-key")
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("wrong key: status %d, want 401", rec.Code)
	}
	rec = postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "secret-key")
	if rec.Code != http.StatusOK {
		t.Errorf("right key: status %d, want 200: %s", rec.Code, rec.Body.String())
	}

	// Health stays public even with keys on.
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	health := httptest.NewRecorder()
	server.routes().ServeHTTP(health, req)
	if health.Code != http.StatusOK {
		t.Errorf("/health with keys on: status %d, want 200", health.Code)
	}
}

func TestNoKeysMeansOpen(t *testing.T) {
	server := testServer(t)
	rec := postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "")
	if rec.Code != http.StatusOK {
		t.Errorf("no keys configured: status %d, want 200", rec.Code)
	}
}

func TestReloadRequiresKeyAndClearsCache(t *testing.T) {
	retriever := testRetriever(t, []string{
		"cats sleep all day in the sun",
		"dogs run in the park every morning",
	})
	// Point the server at a real index file so /reload can re-read it.
	embedder, err := NewEmbedder(256)
	if err != nil {
		t.Fatal(err)
	}
	indexPath := writeTestIndex(t, 256, []Entry{
		{Chunk: testChunk("cats sleep all day in the sun", "d1", nil), Vector: embedder.Embed("cats sleep all day in the sun")},
		{Chunk: testChunk("dogs run in the park every morning", "d1", nil), Vector: embedder.Embed("dogs run in the park every morning")},
	})
	server := NewServer(retriever, indexPath, false, 0)
	server.cache = NewQueryCache(5*time.Minute, 100)
	server.keys = NewAPIKeys([]string{"secret-key"})

	postQuery(t, server, `{"query": "cats sleep", "k": 1}`, "secret-key")
	if entries, _, _ := server.cache.Stats(); entries != 1 {
		t.Fatalf("cache entries = %d, want 1 before reload", entries)
	}

	// Without a key the reload is refused and the cache stays.
	req := httptest.NewRequest(http.MethodPost, "/reload", bytes.NewReader(nil))
	rec := httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("reload without key: status %d, want 401", rec.Code)
	}
	if entries, _, _ := server.cache.Stats(); entries != 1 {
		t.Errorf("cache entries = %d after refused reload, want 1", entries)
	}

	req = httptest.NewRequest(http.MethodPost, "/reload", bytes.NewReader(nil))
	req.Header.Set("X-API-Key", "secret-key")
	rec = httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("reload with key: status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if entries, _, _ := server.cache.Stats(); entries != 0 {
		t.Errorf("cache entries = %d after reload, want 0", entries)
	}
}

func TestNewAPIKeyIsRandom(t *testing.T) {
	a, err := NewAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	if a == b {
		t.Error("two generated keys are identical")
	}
	if len(a) != 64 {
		t.Errorf("key length = %d, want 64 hex chars", len(a))
	}
	keys := NewAPIKeys([]string{a})
	if !keys.Valid(a) || keys.Valid(b) {
		t.Error("generated key did not validate, or a different key did")
	}
	if NewAPIKeys(nil).Enabled() {
		t.Error("empty key set reports enabled")
	}
}
