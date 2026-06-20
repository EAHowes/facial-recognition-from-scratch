# Data Generation

## How to Run

```bash
cd data
python embeddings.py
```

Takes ~3 seconds. Produces `embeddings.npy` and `labels.npy`.

> **Note:** Don't commit the `.npy` files — they're about 98 MB.

## Output Data

| Property | Value |
|---|---|
| Samples | 100,000 |
| Dimensions | 128 per sample |
| Identities | 10 (this is the `k=10`) |
| Samples per identity | 10,000 |
| Labels | Ground truth integers 0–9 |

- All vectors are normalized (length = 1.0)
- Data is pre-shuffled — identities are not grouped together
- Same-identity samples are similar but not identical, like different photos of the same person

## Loading the Data

```python
import numpy as np
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')
```

## Testing Tips

- Start with a small slice: `embeddings[:1000]` and `labels[:1000]`
- Once it works on 1k samples, scale up to 100k
- Seed 42 is fixed, so everyone generates the same data

## Typical Distances

| Pair | Distance |
|---|---|
| Same identity | ~0.7 |
| Different identity | ~1.0 |
