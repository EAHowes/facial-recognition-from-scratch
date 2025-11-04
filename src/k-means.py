# k-means implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer


def compute_distance(data, centriods): # euclidean distance 
    x2 = np.sum(data**2, axis = 1, keepdims=True) # square norm of each data point
    c2 = np.sum(centriods**2, axis = 1) # squared norm of each centriod
    xC = data @ centriods.T # dot product between points and centriods
    dist_squared = x2 + c2 - 2*xC # euclidean dist^2 formula
    return np.sqrt(np.maximum(dist_squared, 0)) # returns actual distance 

def initalize_centriods(data, k):  # assignment 
    first_idx = np.random.randint(data.shape[0]) # random select a centriod 
    centriods = [data[first_idx]] 

    for _ in range(1, k): # selects new centriod each iteration 
        distances = compute_distance(data, np.array(centriods)) # compute distance from each data point from current centriod
        min_dist = np.min(distances, axis = 1)  # finds closet current centriod for each data point
        next_idx = np.argmax(min_dist) # picks point farthest from all current centroid/ this point is new region in data
        centriods.append(data[next_idx]) # new point used for cluster center

    return np.array(centriods)    

