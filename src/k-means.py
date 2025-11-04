# k-means implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer


def compute_distance (data, centriods): # euclidean distance 
    x2 = np.sum(data**2, axis = 1, keepdims=True) # square norm of each data point
    c2 = np.sum(centriods**2, axis = 1) # squared norm of each centriod
    xC = data @ centriods.T # dot product between points and centriods
    dist_squared = x2 + c2 - 2*xC # euclidean dist^2 formula
    return np.sqrt(np.maximum(dist_squared, 0)) # returns actual distance 

def initalize_centriods (data, k):  # initalize/assignment 
    first_idx = np.random.randint(data.shape[0]) # random select a centriod 
    centriods = [data[first_idx]] 

    for _ in range(1, k): # selects new centriod each iteration 
        distances = compute_distance(data, np.array(centriods)) # compute distance from each data point from current centriod
        min_dist = np.min(distances, axis = 1)  # finds closet current centriod for each data point
        next_idx = np.argmax(min_dist) # picks point farthest from all current centroid/ this point is new region in data
        centriods.append(data[next_idx]) # new point used for cluster center

    return np.array(centriods)    

def assign_centriods (data, centriods): # initalize/assignment 
    distances = compute_distance(data, centriods) # compute distance for each centriod 
    labels = np.argmin(distances, axis = 1) # find nearest centriod or each point (min distance)
    min_dists = distances[np.arange(distances.shape[0]), labels] # get minimum distance from each point to its centriod, use for intertia calculation 
    return labels , min_dists

def update_centriods (data, labels, k, old_centroids = None): # updating 
    d = data.shape[1]
    new_centriods = np.zeros((k, d)) # makes centriod maxtrix

    for i in range(k):
        clustered_points = data[labels == i] # points assigned to cluster i 

        if len(clustered_points) > 0:
            # centriod update using the mean of the points inside
            new_centriods[i] = np.mean(clustered_points, axis = 0) 
        else:
           
            if old_centroids is not None: # updating empty clusters 
                
                new_dist = compute_distance(data, old_centroids) #computes distance from all points in old centriod
                assign_dist = new_dist[np.arange(new_dist.shape[0]), labels] # gets distance of each point from its assigned centiord
                far_idx = int(np.argmax(assign_dist)) # finds point farthest from current centriod 
                new_centriods[i] = data[far_idx] # assigns it to a new centriod for cluster i 
            else:
                new_centriods = data[np.random.randomint(data.shape[0])] # no old centriods exist, sets it to a random data
    
    return new_centriods

def compute_inertia(data, labels, centriods): # measures how tight clusters are
    intertia = 0 

