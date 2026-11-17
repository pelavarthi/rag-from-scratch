package main

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
)

// HNSWIndex is an approximate nearest-neighbor index over unit vectors,
// built from scratch following "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs"
// (Malkov & Yashunin, 2018).
//
// The idea: vectors live on a stack of layers. The top layers are sparse
// (a few long-range links), the bottom layer holds every vector. A search
// starts at the top and greedily walks down, so it hops across the graph
// in a few big steps and only scans closely once it is near the answer.
// That is why it is fast: a brute-force scan touches every vector, while
// HNSW touches roughly log(n) of them.
//
// Distance is 1 - cosine similarity. All vectors here are unit length
// (the embedder normalizes them), so cosine similarity is just the dot
// product and a smaller distance means a closer match.
type HNSWIndex struct {
	dim            int
	m              int // max connections per node on layers above 0
	efConstruction int
	entries        []Entry
	nodes          []hnswNode
	entryPoint     int
	maxLevel       int
	rng            *rand.Rand
}

// hnswNode is one vector in the graph. neighbors[l] lists the node ids
// this node links to on layer l; a node on level L has L+1 layers.
type hnswNode struct {
	vector    []float64
	level     int
	neighbors [][]int
}

// HNSWOptions tunes the index. M is the max connections per node on the
// upper layers (layer 0 allows twice that, as in the paper).
// EfConstruction is the beam width used while building: wider means a
// better-connected graph and slower inserts.
type HNSWOptions struct {
	M              int
	EfConstruction int
}

func defaultHNSWOptions() HNSWOptions {
	return HNSWOptions{M: 16, EfConstruction: 64}
}

// NewHNSWIndex creates an empty index for vectors of dim dimensions.
func NewHNSWIndex(dim int, opts HNSWOptions) *HNSWIndex {
	if opts.M <= 0 {
		opts.M = 16
	}
	if opts.EfConstruction <= 0 {
		opts.EfConstruction = 64
	}
	return &HNSWIndex{
		dim:            dim,
		m:              opts.M,
		efConstruction: opts.EfConstruction,
		entryPoint:     -1,
		maxLevel:       -1,
		rng:            rand.New(rand.NewSource(1)),
	}
}

// BuildHNSW indexes every entry in the store.
func BuildHNSW(store *Store, opts HNSWOptions) *HNSWIndex {
	index := NewHNSWIndex(store.Dim, opts)
	for _, entry := range store.entries {
		index.Insert(entry)
	}
	return index
}

// Len returns how many vectors the index holds.
func (h *HNSWIndex) Len() int {
	return len(h.nodes)
}

// MaxLevel is the highest layer in the graph, for tests and stats.
func (h *HNSWIndex) MaxLevel() int {
	return h.maxLevel
}

// distance is 1 - cosine similarity; smaller means closer.
func (h *HNSWIndex) distance(a, b []float64) float64 {
	return 1 - cosine(a, b)
}

// randomLevel draws a node's top layer. Levels fall off geometrically
// (multiplier 1/ln(M)), so most nodes sit on layer 0 and only a few
// reach the top — that is what makes the upper layers sparse.
func (h *HNSWIndex) randomLevel() int {
	ml := 1 / math.Log(float64(h.m))
	return int(-math.Log(h.rng.Float64()) * ml)
}

// Insert adds one entry to the graph.
func (h *HNSWIndex) Insert(entry Entry) {
	level := h.randomLevel()
	id := len(h.nodes)
	h.nodes = append(h.nodes, hnswNode{
		vector:    entry.Vector,
		level:     level,
		neighbors: make([][]int, level+1),
	})
	h.entries = append(h.entries, entry)

	if h.entryPoint == -1 {
		h.entryPoint = id
		h.maxLevel = level
		return
	}

	// Greedy walk down from the entry point to just above this node's
	// top layer: at each layer, keep stepping to the closest neighbor
	// until no neighbor is closer.
	current := h.entryPoint
	currentDist := h.distance(entry.Vector, h.nodes[current].vector)
	for layer := h.maxLevel; layer > level; layer-- {
		for {
			closest, dist := h.closestNeighbor(entry.Vector, current, layer, currentDist)
			if closest == -1 {
				break
			}
			current, currentDist = closest, dist
		}
	}

	// On each layer this node belongs to, run a beam search for close
	// neighbors, link to the best M of them, and continue the walk from
	// the closest one found.
	for layer := min(level, h.maxLevel); layer >= 0; layer-- {
		results := h.searchLayer(entry.Vector, current, h.efConstruction, layer)
		maxConn := h.m
		if layer == 0 {
			maxConn = 2 * h.m
		}
		neighbors := selectNeighbors(results, maxConn)
		for _, neighbor := range neighbors {
			h.link(id, neighbor, layer, maxConn)
		}
		if len(results) > 0 {
			current = results[0].id
		}
	}

	if level > h.maxLevel {
		h.entryPoint = id
		h.maxLevel = level
	}
}

// closestNeighbor returns the neighbor of node on layer that is closer
// to query than dist, or -1 when none is.
func (h *HNSWIndex) closestNeighbor(query []float64, node, layer int, dist float64) (int, float64) {
	best, bestDist := -1, dist
	for _, neighbor := range h.nodes[node].neighbors[layer] {
		if d := h.distance(query, h.nodes[neighbor].vector); d < bestDist {
			best, bestDist = neighbor, d
		}
	}
	return best, bestDist
}

// link connects a and b on layer, pruning b's list to maxConn closest
// neighbors when it overflows.
func (h *HNSWIndex) link(a, b, layer, maxConn int) {
	h.nodes[a].neighbors[layer] = append(h.nodes[a].neighbors[layer], b)
	list := append(h.nodes[b].neighbors[layer], a)
	if len(list) <= maxConn {
		h.nodes[b].neighbors[layer] = list
		return
	}
	scored := make([]candidate, 0, len(list))
	for _, id := range list {
		scored = append(scored, candidate{
			id:   id,
			dist: h.distance(h.nodes[b].vector, h.nodes[id].vector),
		})
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].dist < scored[j].dist })
	pruned := make([]int, 0, maxConn)
	for _, c := range scored[:maxConn] {
		pruned = append(pruned, c.id)
	}
	h.nodes[b].neighbors[layer] = pruned
}

// selectNeighbors keeps the maxConn closest candidates.
func selectNeighbors(results []candidate, maxConn int) []int {
	if len(results) > maxConn {
		results = results[:maxConn]
	}
	ids := make([]int, 0, len(results))
	for _, c := range results {
		ids = append(ids, c.id)
	}
	return ids
}

// Search returns the k entries closest to query, best first. ef is the
// beam width of the layer-0 search: wider is more accurate and slower.
// With no filter it is approximate; callers needing exact results or a
// metadata filter should use the brute-force store search instead.
func (h *HNSWIndex) Search(query []float64, k, ef int) []ScoredEntry {
	if len(h.nodes) == 0 || k <= 0 {
		return nil
	}
	if ef < k {
		ef = k
	}

	// Greedy descent through the upper layers, one closest step at a
	// time, to arrive near the answer before the real search starts.
	current := h.entryPoint
	currentDist := h.distance(query, h.nodes[current].vector)
	for layer := h.maxLevel; layer > 0; layer-- {
		for {
			closest, dist := h.closestNeighbor(query, current, layer, currentDist)
			if closest == -1 {
				break
			}
			current, currentDist = closest, dist
		}
	}

	results := h.searchLayer(query, current, ef, 0)
	if k > len(results) {
		k = len(results)
	}
	hits := make([]ScoredEntry, 0, k)
	for _, c := range results[:k] {
		entry := h.entries[c.id]
		hits = append(hits, ScoredEntry{
			Chunk:  entry.Chunk,
			Score:  1 - c.dist,
			Vector: entry.Vector,
		})
	}
	return hits
}

// candidate is one node found by a beam search, with its distance to
// the query.
type candidate struct {
	id   int
	dist float64
}

// searchLayer runs a beam search from entry on one layer and returns the
// ef closest nodes found, best first.
func (h *HNSWIndex) searchLayer(query []float64, entry, ef, layer int) []candidate {
	visited := map[int]bool{entry: true}
	frontier := &minHeap{}
	results := &maxHeap{}
	start := candidate{id: entry, dist: h.distance(query, h.nodes[entry].vector)}
	heap.Push(frontier, start)
	heap.Push(results, start)

	for frontier.Len() > 0 {
		// Stop once the closest unvisited node is farther than the
		// worst result kept so far: nothing beyond it can improve
		// the answer.
		closest := (*frontier)[0]
		if results.Len() >= ef && closest.dist > (*results)[0].dist {
			break
		}
		current := heap.Pop(frontier).(candidate)
		for _, neighbor := range h.nodes[current.id].neighbors[layer] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			found := candidate{
				id:   neighbor,
				dist: h.distance(query, h.nodes[neighbor].vector),
			}
			if results.Len() < ef || found.dist < (*results)[0].dist {
				heap.Push(frontier, found)
				heap.Push(results, found)
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	found := make([]candidate, 0, results.Len())
	for results.Len() > 0 {
		found = append(found, heap.Pop(results).(candidate))
	}
	sort.Slice(found, func(i, j int) bool { return found[i].dist < found[j].dist })
	return found
}

// minHeap is a closest-first priority queue of candidates.
type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	item := old[len(old)-1]
	*h = old[:len(old)-1]
	return item
}

// maxHeap is a farthest-first priority queue of candidates, so the
// worst result kept is always on top.
type maxHeap []candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	item := old[len(old)-1]
	*h = old[:len(old)-1]
	return item
}
