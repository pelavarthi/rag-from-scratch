package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testServer(t *testing.T) *Server {
	t.Helper()
	retriever := testRetriever(t, []string{
		"cats sleep all day in the sun",
		"dogs run in the park every morning",
	})
	return NewServer(retriever, "")
}

func TestQueryEndpoint(t *testing.T) {
	server := testServer(t)
	body := bytes.NewBufferString(`{"query": "cats sleep", "k": 1}`)
	req := httptest.NewRequest(http.MethodPost, "/query", body)
	rec := httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var resp queryResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(resp.Results))
	}
	if resp.Results[0].Text != "cats sleep all day in the sun" {
		t.Errorf("first hit = %q", resp.Results[0].Text)
	}
	if resp.Results[0].Score <= 0 {
		t.Errorf("score = %v, want positive", resp.Results[0].Score)
	}
}

func TestQueryEndpointRejectsEmptyQuery(t *testing.T) {
	server := testServer(t)
	req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(`{"query": ""}`))
	rec := httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestQueryEndpointRejectsBadJSON(t *testing.T) {
	server := testServer(t)
	req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(`{oops`))
	rec := httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestQueryEndpointRejectsGet(t *testing.T) {
	server := testServer(t)
	req := httptest.NewRequest(http.MethodGet, "/query", nil)
	rec := httptest.NewRecorder()
	server.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestHealthAndStats(t *testing.T) {
	server := testServer(t)
	for _, path := range []string{"/health", "/stats"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		server.routes().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d, want 200", path, rec.Code)
		}
		var body map[string]any
		if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["chunks"] != float64(2) {
			t.Errorf("%s chunks = %v, want 2", path, body["chunks"])
		}
	}
}
