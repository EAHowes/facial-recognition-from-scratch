# DBSCAN implementation

import numpy as np

def dbscan(dataSet, eps, minPts):
    # Scans all unvisited points and labels them accordingly
    # Core DBSCAN functionality implemented here with grow_cluster and region_query helper functions below

    labels_out = [0] * len(dataSet)   # initialize all points to unvisited

    current_cluster = 0

    for point in range(0, len(dataSet)):
        if (labels_out[point] != 0):    # only check unvisited
            continue

        neighborPts = region_query(dataSet, point, eps)     # check if core point or not
        if len(neighborPts) < minPts:   # noise
            labels_out[point] = -1
        else:
            current_cluster += 1    # core
            grow_cluster(dataSet, labels_out, point, neighborPts, current_cluster, eps, minPts)     # add point and all surrounding points to cluster

    return labels_out


def grow_cluster(dataSet, labels_out, point, neighborPts, current_cluster, eps, minPts):
    # After finding a core point, this finds all other core points and boarder points to add to the cluster

    labels_out[point] = current_cluster     # initialize a new cluster for the new core point

    i = 0

    queued = set(neighborPts)   # prevent duplicat points in search

    # loop acts similarly to a BFS search
    while i < len(neighborPts):
        next_point = neighborPts[i]

        if labels_out[next_point] == -1:    # relabel point if noise
            labels_out[next_point] = current_cluster

        elif labels_out[next_point] == 0:   # add unvisited point and all neightbors if its a core point 
            labels_out[next_point] = current_cluster

            next_neighbors = region_query(dataSet, next_point, eps)
            if len(next_neighbors) >= minPts:
                for n in next_neighbors:
                    if n not in queued:
                        neighborPts.append(n)
                        queued.add(n)

        i += 1

def region_query(dataSet, point, eps):
    # Finds points that are within a certain distance (eps) of the given point
    # Uses a vectorized method of finding the euclidian distance of cells ... vectorization significantly reduces runtime

    diff = dataSet - dataSet[point]             # creates a matrix of coordinate differences
    dist_sq = np.sum(diff * diff, axis = 1)     # square everything for euclidian distance
    mask = dist_sq <= eps * eps                 # compare to eps^2 instead of taking sqrt for better runtime
    neighbors = np.where(mask)[0].tolist()      # save indicies of neightbors

    return neighbors
