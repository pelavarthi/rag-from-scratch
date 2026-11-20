# HNSW tuning report

Synthetic index: 2000 vectors, dim 256, 3.9 MB of vectors. Sweep of M in {8, 16, 32} x search ef in {16, 32, 64, 128}; each M is built once and searched at every ef, as the server does. 50 queries, k=10, run in 7.38932362s.

```
HNSW tuning: 12 settings, recall@10 vs brute force
setting             build        p50        p99   recall      edges       size
 brute force           0s 1.174456ms 3.300452ms    1.000          0     3.9 MB
 M=8 ef=16     993.964858ms  161.282µs  373.542µs    0.382      34252     4.2 MB
 M=8 ef=32     993.964858ms  270.737µs  325.379µs    0.542      34252     4.2 MB
 M=8 ef=64     993.964858ms  459.291µs  793.904µs    0.722      34252     4.2 MB
 M=8 ef=128    993.964858ms  753.703µs 1.261096ms    0.870      34252     4.2 MB
 M=16 ef=16    1.785577962s  292.389µs  460.112µs    0.572      65946     4.4 MB
 M=16 ef=32    1.785577962s  443.757µs  802.216µs    0.762      65946     4.4 MB
 M=16 ef=64    1.785577962s  680.463µs  981.476µs    0.930      65946     4.4 MB
*M=16 ef=128   1.785577962s  1.03726ms 1.474747ms    0.982      65946     4.4 MB
 M=32 ef=16    4.095603652s  442.876µs 1.218131ms    0.760     130402     4.9 MB
 M=32 ef=32    4.095603652s  661.434µs 1.136148ms    0.924     130402     4.9 MB
 M=32 ef=64    4.095603652s  972.772µs 1.803101ms    0.992     130402     4.9 MB
 M=32 ef=128   4.095603652s 1.389679ms 1.972446ms    0.998     130402     4.9 MB
recommended: M=16 ef=128 (recall 0.982, p99 1.474747ms)
```

Note: random vectors are near-orthogonal, so these recall numbers are a pessimistic bound — real text embeddings cluster, and recall on those should be higher at the same ef. At this small scale HNSW barely beats brute force; the graph pays off as the index grows. Defaults stay M=16, ef=64 until measured on real data.
