# K-Means (Cosine Distance)

## Files

| File | Purpose |
|---|---|
| `k_means.py` | Core algorithm implementation |
| `k_means_wrapper.py` | Loads data, configures parameters, runs K-Means, prints results |

Both files share the same time complexity: **O(n × k × d × i)**

| Variable | Meaning |
|---|---|
| `n` | number of samples |
| `k` | number of clusters |
| `d` | dimensions per data point |
| `i` | iterations until convergence |

## How to Run

```bash
cd src/
python -m src.k_means_wrapper
```

## Output

```
Running K-Means with K=10
--- K-Means Results ---
    Number of iterations 3
    runtime: 0.15 seconds
    centroids shape: (10, 128)
    Inertia: 4930.976106084883
    Cluster counts: [1000 1000 1000 1000 1000 1000 1000 1000 1000 1000]
    First 10 labels: [2 3 5 2 0 1 7 1 3 9]
```

Fields printed to console:

- `Number of iterations` — how many iterations were performed before convergence
- `runtime` — execution time in seconds
- `centroids shape` — `(number of clusters, dimensions)`
- `Inertia` — clustering compactness: sum of minimum cosine distances
- `Cluster counts` — number of samples assigned to each cluster after convergence
- `First 10 labels` — cluster label assigned to the first 10 samples (labels range 0–9, K=10)
