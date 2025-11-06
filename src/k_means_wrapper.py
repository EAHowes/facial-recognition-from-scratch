import numpy as np
import time
from src.k_means import Kmeans

if __name__ == "__main__":
    # load embeddings and labels
    embeddings = np.load("data/embeddings.npy")
    labels = np.load("data/labels.npy")

    K = 10
    print(f"Running K-Means with K={K}...")

    start = time.time()
    km = Kmeans(K=K, max_iters= 5)
    km.fit_data(embeddings)
    end = time.time()

    print("\n--- K-Means Results ---")
    print(f"Number of iterations {km.n_iter_}")
    print(f"runtime: {end - start:.2f} seconds")
    print(f"centroids shape: {km.centroids.shape}")
    print(f"Inertia: {km.inertia}")
    print(f"Cluster counts: {np.bincount(km.labels, minlength=K)}")
    print(f"First 10 labels: {km.labels[:10]}")

# python -m src.k_means_wrapper