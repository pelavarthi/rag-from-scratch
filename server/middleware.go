// Caching, rate limiting, and API-key auth for the query API.
//
// The three pieces protect the same endpoint in different ways: the
// cache skips repeat work, the limiter keeps one caller from hogging
// the server, and API keys decide who may call it at all. All three
// are stdlib-only and sit in front of the retriever, never inside it.
package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"sync"
	"time"
)

// QueryCache remembers recent query responses. The key is the whole
// request (query text, k, MMR settings, filters), so two identical
// requests share one retrieval and anything different misses.
type QueryCache struct {
	mu      sync.Mutex
	ttl     time.Duration
	max     int
	entries map[string]cacheEntry
	order   []string // insertion order, oldest first, for eviction
	hits    int
	misses  int
}

type cacheEntry struct {
	hits     []queryHit
	inserted time.Time
}

// NewQueryCache makes a cache that keeps at most max responses for ttl.
func NewQueryCache(ttl time.Duration, max int) *QueryCache {
	return &QueryCache{
		ttl:     ttl,
		max:     max,
		entries: make(map[string]cacheEntry),
	}
}

func cacheKey(req queryRequest) string {
	// json.Marshal sorts map keys, so equal filters produce equal keys.
	raw, err := json.Marshal(req)
	if err != nil {
		return req.Query
	}
	return string(raw)
}

// Get returns the cached hits for key, or false on a miss or an
// entry older than the TTL.
func (c *QueryCache) Get(key string) ([]queryHit, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok {
		c.misses++
		return nil, false
	}
	if time.Since(entry.inserted) > c.ttl {
		c.remove(key)
		c.misses++
		return nil, false
	}
	c.hits++
	return entry.hits, true
}

// Put stores hits under key, evicting the oldest entries when full.
// A repeat Put refreshes the entry in place instead of duplicating it.
func (c *QueryCache) Put(key string, hits []queryHit) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.entries[key]; !ok {
		for len(c.entries) >= c.max && len(c.order) > 0 {
			c.remove(c.order[0])
		}
		c.order = append(c.order, key)
	}
	c.entries[key] = cacheEntry{hits: hits, inserted: time.Now()}
}

// Clear drops every entry, e.g. after /reload swaps in a new index.
func (c *QueryCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[string]cacheEntry)
	c.order = nil
}

// Stats reports the live entry count plus lifetime hits and misses.
func (c *QueryCache) Stats() (entries, hits, misses int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries), c.hits, c.misses
}

func (c *QueryCache) remove(key string) {
	delete(c.entries, key)
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

// RateLimiter is a token bucket per caller: each caller starts with
// burst tokens and earns rate more per second, so short bursts pass
// but a sustained flood is throttled.
type RateLimiter struct {
	mu      sync.Mutex
	rate    float64 // tokens refilled per second
	burst   int
	buckets map[string]*bucket
}

type bucket struct {
	tokens  float64
	updated time.Time
}

// NewRateLimiter makes a limiter allowing rate requests per second
// with bursts of up to burst at once.
func NewRateLimiter(rate float64, burst int) *RateLimiter {
	return &RateLimiter{rate: rate, burst: burst, buckets: make(map[string]*bucket)}
}

// Allow spends one token for caller and reports whether one was left.
// A caller with no bucket starts full, so the first burst always passes.
func (l *RateLimiter) Allow(caller string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	b, ok := l.buckets[caller]
	if !ok {
		b = &bucket{tokens: float64(l.burst), updated: now}
		l.buckets[caller] = b
	}
	b.tokens += l.rate * now.Sub(b.updated).Seconds()
	if b.tokens > float64(l.burst) {
		b.tokens = float64(l.burst)
	}
	b.updated = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// APIKeys holds the set of accepted keys. Only SHA-256 hashes are
// stored, so a config dump or a log line never reveals a live key.
type APIKeys struct {
	hashes map[string]bool
}

// NewAPIKeys builds the set from raw keys, ignoring blank entries.
func NewAPIKeys(raw []string) *APIKeys {
	hashes := make(map[string]bool)
	for _, key := range raw {
		if key == "" {
			continue
		}
		hashes[hashKey(key)] = true
	}
	return &APIKeys{hashes: hashes}
}

func hashKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// Enabled reports whether any key is configured. With no keys the
// server runs open, as before; with keys, /query and /reload require one.
func (k *APIKeys) Enabled() bool { return len(k.hashes) > 0 }

// Valid reports whether key is one of the configured keys.
func (k *APIKeys) Valid(key string) bool {
	if key == "" {
		return false
	}
	return k.hashes[hashKey(key)]
}

// NewAPIKey mints a fresh random key: 32 bytes, hex-encoded.
func NewAPIKey() (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return hex.EncodeToString(raw), nil
}

// callerID identifies a request for rate limiting: the API key when
// one was presented, otherwise the client IP.
func callerID(r *http.Request) string {
	if key := r.Header.Get("X-API-Key"); key != "" {
		return "key:" + hashKey(key)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return "ip:" + host
}

// requireKey rejects requests without a valid API key when keys are
// configured, and passes everything through when they are not.
func (s *Server) requireKey(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.keys != nil && s.keys.Enabled() && !s.keys.Valid(r.Header.Get("X-API-Key")) {
			writeError(w, http.StatusUnauthorized, "missing or invalid X-API-Key header")
			return
		}
		next(w, r)
	}
}

// rateLimit rejects a request once its caller has spent its tokens,
// with a Retry-After hint of one second.
func (s *Server) rateLimit(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.limiter != nil && !s.limiter.Allow(callerID(r)) {
			w.Header().Set("Retry-After", "1")
			writeError(w, http.StatusTooManyRequests, "rate limit exceeded, try again shortly")
			return
		}
		next(w, r)
	}
}
