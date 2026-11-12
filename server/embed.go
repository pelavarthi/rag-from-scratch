package main

import (
	"crypto/md5"
	"encoding/binary"
	"errors"
	"math"
	"regexp"
	"strings"
)

// wordPattern splits text into words. Python's \w also matches Unicode
// letters; this ASCII-only pattern matches it for the English text this
// project indexes.
var wordPattern = regexp.MustCompile(`\w+`)

// Embedder turns text into a fixed-length vector. It is a port of the
// Python HashingEmbedder, so a query embedded here lands in the same
// space as chunk vectors built by the Python pipeline.
type Embedder struct {
	Dim int
}

func NewEmbedder(dim int) (*Embedder, error) {
	if dim <= 0 {
		return nil, errors.New("dim must be positive")
	}
	return &Embedder{Dim: dim}, nil
}

// Embed lowercases the text, hashes each word with MD5 into one of Dim
// buckets, counts hits per bucket, and L2-normalizes the result.
func (e *Embedder) Embed(text string) []float64 {
	vector := make([]float64, e.Dim)
	for _, word := range wordPattern.FindAllString(strings.ToLower(text), -1) {
		sum := md5.Sum([]byte(word))
		bucket := int(binary.BigEndian.Uint32(sum[:4])) % e.Dim
		vector[bucket]++
	}
	return normalize(vector)
}

func normalize(vector []float64) []float64 {
	length := 0.0
	for _, x := range vector {
		length += x * x
	}
	length = math.Sqrt(length)
	if length == 0 {
		return vector
	}
	for i := range vector {
		vector[i] /= length
	}
	return vector
}

// cosine is the dot product. It assumes unit-length vectors, which is
// what Embedder returns and what the saved index holds.
func cosine(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
