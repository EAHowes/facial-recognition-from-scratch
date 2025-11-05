# k-means implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
"""""
def compute_distance(data, centroids): #cosine distance 
   
    # Normalize data and centroids to unit vectors
    data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-12)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    # cosine similarity
    similarity = data_norm @ cent_norm.T  # dot product
    # convert to cosine distance
    distance = 1 - similarity
    # clip numerical floating point errors to keep valid range
    return np.clip(distance, 0, 2)
"""""
def compute_distance (data, centroids): # euclidean distance 
    x2 = np.sum(data**2, axis = 1, keepdims=True) # square norm of each data point
    c2 = np.sum(centroids**2, axis = 1) # squared norm of each centriod
    xC = data @ centroids.T # dot product between points and centriods
    dist_squared = x2 + c2 - 2*xC # euclidean dist^2 formula
    return np.sqrt(np.maximum(dist_squared, 0)) # returns actual distance 

def initalize_centroids (data, k):  # initalize/assignment 
    first_idx = np.random.randint(data.shape[0]) # random select a centriod 
    centroids = [data[first_idx]] 

    for _ in range(1, k): # selects new centriod each iteration 
        distances = compute_distance(data, np.array(centroids)) # compute distance from each data point from current centriod
        min_dist = np.min(distances, axis = 1)  # finds closet current centriod for each data point
        next_idx = np.argmax(min_dist) # picks point farthest from all current centroid/ this point is new region in data
        centroids.append(data[next_idx]) # new point used for cluster center

    return np.array(centroids)    

def assign_centroids (data, centroids): # initalize/assignment 
    distances = compute_distance(data, centroids) # compute distance for each centriod 
    labels = np.argmin(distances, axis = 1) # find nearest centriod or each point (min distance)
    min_dists = distances[np.arange(distances.shape[0]), labels] # get minimum distance from each point to its centriod, use for intertia calculation 
    return labels , min_dists

def update_centroids (data, labels, k, old_centroids = None): # updating 
    d = data.shape[1]
    new_centroids = np.zeros((k, d)) # makes centriod maxtrix

    for i in range(k):
        clustered_points = data[labels == i] # points assigned to cluster i 

        if len(clustered_points) > 0:
            # centriod update using the mean of the points inside
            new_centroids[i] = np.mean(clustered_points, axis = 0) 
        else:
           
            if old_centroids is not None: # updating empty clusters 
                
                new_dist = compute_distance(data, old_centroids) #computes distance from all points in old centriod
                assign_dist = new_dist[np.arange(new_dist.shape[0]), labels] # gets distance of each point from its assigned centiord
                far_idx = int(np.argmax(assign_dist)) # finds point farthest from current centriod 
                new_centroids[i] = data[far_idx] # assigns it to a new centriod for cluster i 
            else:
                new_centroids = data[np.random.randomint(data.shape[0])] # no old centriods exist, sets it to a random data
    
    return new_centroids

def compute_inertia(min_dists): # measures how tight clusters are
    inertia = np.sum(min_dists**2) # min_dist find distance from point i to the centroid it belongs then we square and sum them up
    return float(inertia)
    

class Kmeans:

    def __init__(self, K=10, max_iters = 20, plot_steps = False, tol = 1e-4):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit_data(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape 

        self.centriods = initalize_centroids(X, self.K) #initlizae K centroids

        for _ in range(self.max_iters):
            dist = compute_distance(X, self.centroids) #compute distance from each point to each centroid
            labels= np.argmin(dist, axis = 1) 
            #this below might be wrong 
            updated_centroids = update_centroids(X,labels,self.K, old_centroids= self.centriods)
            convergence = np.linalg.norm (updated_centroids - self.centroids)
            self.centroids = update_centroids
            self.labels = labels

            if convergence <= self.tol:
                break
                
            self.inertia = compute_inertia(dist)
            return self 






