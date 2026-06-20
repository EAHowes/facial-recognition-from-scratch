# DBSCAN

This implementation works correctly but is slow by nature — expect **30–90 minutes** depending on your machine (80 min on a laptop, ~30 min on a desktop). The bottleneck is CPU-bound NumPy distance computation, which can't easily be parallelized.

## Files

| File | Complexity | Purpose |
|---|---|---|
| `dbscan.py` | O(n² × d) | Core algorithm: finds core points, grows clusters via `region_query` using NumPy euclidean distance |
| `dbscan_wrapper.py` | O(n² × d) | Loads embeddings, sets `eps` and `minPts`, calls DBSCAN, prints results |

## How to Run

```bash
cd src/
python dbscan_wrapper.py
```

## Output

```
dbscan results
eps: 1.007
minPts: 10
runtime: 2783.64 s
points: 100000
clusters: 10
noise points: 28752
Purity: 1.0000  (based on 71248 non-noise points)
```

Fields printed to console:

- `eps` — neighborhood radius
- `minPts` — minimum neighbors to form a core point
- `runtime` — total execution time in seconds
- `points` — total number of input points
- `clusters` — number of clusters found
- `noise points` — points not assigned to any cluster
- `Purity` — accuracy computed over non-noise points only
