# This file is used to run the dbscan implementation in dbscan.py 
import time 
import numpy as np
from pathlib import Path 
from dbscan import dbscan

if __name__ == "__main__":
    # Final values for optimized dbscan
    EPS = 1.007
    MINPTS = 10

    EMBEDDINGS_ARRAY = np.load(Path("../data/embeddings.npy"))
    EMBEDDINGS_ARRAY = EMBEDDINGS_ARRAY.astype(np.float32, copy=False)
 
    ############################################
    # eps optimization code
    # uses only the first SAMPLE_SIZE number of points to test clustering

    # Only test on the first 10,000 points
    # SAMPLE_SIZE = 10000
    # EMBEDDINGS_ARRAY = EMBEDDINGS_ARRAY[:SAMPLE_SIZE]

    # initialize eps testing range
    # for eps in [1.006, 1.007, 1.008, 1.009]:

    t0 = time.time()
    labels_out = dbscan(EMBEDDINGS_ARRAY, EPS, MINPTS)
    time_passed = time.time() - t0

    predicted_clustering = np.asarray(labels_out, dtype=int)

    points = len(predicted_clustering)
    noise_points = int(np.sum(predicted_clustering == -1))
    cluster_ids = [c for c in np.unique(predicted_clustering) if c >= 1]    # sum cluters by ids
    clusters = len(cluster_ids)

    print("\ndbscan results")
    print(f"eps: {EPS}")
    print(f"minPts: {MINPTS}")
    print(f"runtime: {time_passed:.2f} s")
    print(f"points: {points}")
    print(f"clusters: {clusters}")
    print(f"noise points: {noise_points}")

    # compare the predicted results from home-made dbscan to the ground truth labels
    GROUND_TRUTH = np.load(Path("../data/labels.npy"))

    # Make ground truth size the same as embedding size in case of limiting SAMPLE_SIZE
    GROUND_TRUTH = GROUND_TRUTH[:SAMPLE_SIZE]
    assert len(labels_out) == len(GROUND_TRUTH)

   # ignore all noise points
    no_noise_clusters = predicted_clustering != -1      # Find where all the noise points are
    predicted_clustering = predicted_clustering[no_noise_clusters]      # remove noise points from prediction and gt
    gt = GROUND_TRUTH[no_noise_clusters]

    if len(predicted_clustering) == 0:
        print("Error: all points were labeled as noise. cannot compute accuracy")
    else:
        total = len(predicted_clustering)   # find total numbers of non noise points
        correct = 0
        for cluster_id in np.unique(predicted_clustering):      # Loop through each cluster
            idx = (predicted_clustering == cluster_id)      # for each point in cluster_id
            labels, counts = np.unique(gt[idx], return_counts=True)     # using the idx mask, remove all the non ground truth vals
            correct += counts.max()     # taking the majority count for each cluster
        clustering_accuracy = correct / total
        print(f"clustering accuracy: {clustering_accuracy:.2f}")
