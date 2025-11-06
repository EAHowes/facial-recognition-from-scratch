import numpy as np
from k_means import Kmeans

# X: shape (n_samples, n_features), e.g., your 128-D face embeddings
# If using cosine, pre-normalization helps but isn't required since the class normalizes internally.
# X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

k = 10
km = Kmeans(K=k, max_iters=200, tol=1e-4)
print(km)
#km.fit_data(km)

print("Centroids:", km.centroids)
print("Inertia:", km.inertia)
print("Cluster counts:")
