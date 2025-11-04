# This file is used to run the dbscan implementation in dbscan.py 
import os
import multiprocessing

# Detect total logical CPU cores
_total_cores = multiprocessing.cpu_count()

# Leave 2 cores free (at least 1 core always used)
_use_cores = max(1, _total_cores - 3)

# Apply to FlexiBLAS and OpenMP environments
os.environ["FLEXIBLAS_NUM_THREADS"] = str(_use_cores)
os.environ["OMP_NUM_THREADS"] = str(_use_cores)

# Optional: disable other BLAS vars to avoid oversubscription
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.pop(var, None)

print(f"[FlexiBLAS/OpenBLAS] Using {_use_cores}/{_total_cores} threads for NumPy operations.")

import time 
import numpy as np
from pathlib import Path 
from dbscan import dbscan

if __name__ == "__main__":
    # Final implementation of dbscan with optimized eps and minPts values
    # EPS = 1.15
    MINPTS = 10

    EMBEDDINGS_ARRAY = np.load(Path("../data/embeddings.npy"))
    EMBEDDINGS_ARRAY = EMBEDDINGS_ARRAY.astype(np.float32, copy=False)
 
############################################
# eps optimization code

    # Only test on the first 10,000 points
    # LIMIT = 10000
    # if EMBEDDINGS_ARRAY.shape[0] > LIMIT:
    #     EMBEDDINGS_ARRAY = EMBEDDINGS_ARRAY[:LIMIT]

    # initialize eps testing range
    for eps in [1.0, 1.01, 1.02, 1.03]:

        t0 = time.time()
        labels_out = dbscan(EMBEDDINGS_ARRAY, eps, MINPTS)
        time_passed = time.time() - t0

        predicted_clustering = np.asarray(labels_out, dtype=int)

        points = len(predicted_clustering)
        noise_points = int(np.sum(predicted_clustering == -1))
        cluster_ids = [c for c in np.unique(predicted_clustering) if c >= 1]
        clusters = len(cluster_ids)

        print("\ndbscan results")
        print(f"eps: {eps}")
        print(f"runtime: {time_passed:.2f} s")
        print(f"points: {points}")
        print(f"clusters: {clusters}")
        print(f"noise points: {noise_points}")

        # compare the predicted results from home-made dbscan to the ground truth labels
        GROUND_TRUTH = np.load(Path("../data/labels.npy"))

        # make ground truth size the same as embedding size
        GROUND_TRUTH = GROUND_TRUTH[:EMBEDDINGS_ARRAY.shape[0]]
        assert len(labels_out) == len(GROUND_TRUTH)

        # ignore all noise points
        mask = predicted_clustering != -1
        predicted_clustering = predicted_clustering[mask]
        gt = GROUND_TRUTH[mask]

        if len(predicted_clustering) == 0:
            print("no points left. all points were labeled as noise.")
        else:
            total = len(predicted_clustering)   # find total numbers of non noise points
            correct = 0
            for cluster_id in np.unique(predicted_clustering):      # compute the correctness of each point found in cluster
                idx = (predicted_clustering == cluster_id)
                labels, counts = np.unique(gt[idx], return_counts=True)
                correct += counts.max()
            purity = correct / total
            print(f"Purity: {purity:.4f}  (based on {total} non-noise points)")

############################################

    # t0 = time.time()
    # labels_out = dbscan(EMBEDDINGS_ARRAY, EPS, MINPTS)
    # time_passed = time.time() - t0
    #
    # predicted_clustering = np.asarray(labels_out, dtype=int)
    #
    # points = len(predicted_clustering)
    # noise_points = int(np.sum(predicted_clustering == -1))
    # cluster_ids = [c for c in np.unique(predicted_clustering) if c >= 1]
    # clusters = len(cluster_ids)
    #
    # print("\ndbscan results")
    # print(f"runtime: {time_passed:.2f} s")
    # print(f"points: {points}")
    # print(f"clusters: {clusters}")
    # print(f"noise points: {noise_points}")
    #
    # GROUND_TRUTH = np.load(Path("../data/labels.npy"))
    #
    # if GROUND_TRUTH.shape[0] > EMBEDDINGS_ARRAY.shape[0]:
    #     GROUND_TRUTH = GROUND_TRUTH[:EMBEDDINGS_ARRAY.shape[0]]
    #
    # assert len(labels_out) == len(GROUND_TRUTH)
    #
    # mask = predicted_clustering != -1
    # predicted_clustering = predicted_clustering[mask]
    # gt = GROUND_TRUTH[mask]
    #
    # if len(predicted_clustering) == 0:
    #     print("All points were labeled as noise — no purity score.")
    # else:
    #     total = len(predicted_clustering)
    #     correct = 0
    #     for cluster_id in np.unique(predicted_clustering):
    #         idx = (predicted_clustering == cluster_id)
    #         # find the majority ground-truth label in this cluster
    #         labels, counts = np.unique(gt[idx], return_counts=True)
    #         correct += counts.max()
    #     purity = correct / total
    #     print(f"Purity: {purity:.4f}  (based on {total} non-noise points)")
    #
