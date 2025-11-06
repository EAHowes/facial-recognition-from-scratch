# k-means implementation
import numpy as np
import matplotlib.pyplot as plt

""""
def compute_distance (data, centroids): # euclidean distance 
    x2 = np.sum(data**2, axis = 1, keepdims=True) # square norm of each data point
    c2 = np.sum(centroids**2, axis = 1) # squared norm of each centriod
    xC = data @ centroids.T # dot product between points and centriods
    dist_squared = x2 + c2 - 2*xC # euclidean dist^2 formula
    return np.sqrt(np.maximum(dist_squared, 0)) # returns actual distance 
"""

def compute_distance(data, centroids): #cosine distance 
    # Normalize data and centroids to unit vectors
    data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-12)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    cos_similarity = data_norm @ cent_norm.T  # dot product
    distance = 1 - cos_similarity             # convert to cosine distance
    return np.clip(distance, 0, 2)            # clip numerical floating point errors to keep valid range

def initalize_centroids (data, k):  # initalize/assignment 
    rng = np.random.default_rng(42)
    rand_idx = rng.integers(data.shape[0]) # random select a centriod 
    centroids = [data[rand_idx]] 

    for _ in range(1, k): # selects new centriod each iteration 
        distances = compute_distance(data, np.array(centroids)) # compute distance from each data point from current centriod (matrix shape (n_samples, curr_num_centroids))
        min_dist = np.min(distances, axis = 1)  # finds closet current centriod for each data point
        next_idx = np.argmax(min_dist) # picks point farthest from all current centroid/ this point is new region in data
        centroids.append(data[next_idx]) # new point used for cluster center

    return np.array(centroids)    

def assign_centroids (data, centroids): # initalize/assignment 
    distances = compute_distance(data, centroids) # compute distance for each centriod, matrix shape:(n_samples, k), distances[i,j] = distance from point i to j
    labels = np.argmin(distances, axis = 1) # find nearest centriod or each point (min distance), maps each row(sample) to it's cluster 
    min_dists = np.min(distances, axis = 1) # get minimum distance from each point to its centriod, use for intertia calculation 
    return labels , min_dists

def update_centroids (data, labels, k, old_centroids = None, rng = None): # updating 
    """
    data: (n, d)
    labels: (n,) ints in [0, k-1]
    k: number of clusters
    old_centroids: (k, d) or None   
    rng: np.random.Generator or None
    """

    n_samples,d = data.shape         # (samples, features)
    new_centroids = np.zeros((k, d)) # makes centriod maxtrix

    if rng is None:
     rng = np.random.default_rng()

    for i in range(k):
        clustered_points = data[labels == i] # points assigned to cluster i 
        
         
        if len(clustered_points) > 0:
            # centriod update using the mean of the points inside
            new_centroids[i] = np.mean(clustered_points, axis = 0) 
            new_centroids[i] /= np.linalg.norm(new_centroids[i]) + 1e-12 #norm for cosine distance 
        else:
           
            if old_centroids is not None: # updating empty clusters 
                
                new_dist = compute_distance(data, old_centroids) # computes distance from all points in old centriod
                assign_dist = new_dist[np.arange(new_dist.shape[0]), labels] # gets distance of each point from its assigned centiord
                far_idx = int(np.argmax(assign_dist)) # finds point farthest from current centriod 
                new_centroids[i] = data[far_idx]      # assigns it to a new centriod for cluster i 

                new_centroids[i] /= np.linalg.norm(new_centroids[i]) + 1e-12 # norm for cosine distance
            else:
                new_centroids[i] = data[np.random.randint(data.shape[0])]    # no old centriods exist, sets it to a random data

                new_centroids[i] /= np.linalg.norm(new_centroids[i]) + 1e-12 # norm for cosine distance
    
    return new_centroids

def compute_inertia(min_dists): # measures how tight clusters are
    #inertia = np.sum(min_dists**2) # min_dist find distance from point i to the centroid it belongs then we square and sum them up
    #return float(inertia)
    return float(np.sum(min_dists)) # for cosine distance 
    

class Kmeans:

    def __init__(self, K=12, max_iters = 20,tol = 1e-4):
        self.K = K                 # number of cluster
        self.max_iters = max_iters # max k means iterations 
        self.tol = tol             # convergence tolerance (meaning the clusters arent changing)
        self.centroids = None      # store learned centroids
        self.labels = None         # stores final cluster labels
        self.inertia = None        # stores final intertia(how tight clusters are)

    def fit_data(self, X):
        self.X = X
        n_samples, self.n_features = X.shape # row and column data points in the dataset

        self.centroids = initalize_centroids(X, self.K) # initlizae K centroids (init)

        for _ in range(self.max_iters):
            # assigning 
            dist = compute_distance(X, self.centroids) # compute distance from each point to each centroid
            labels= np.argmin(dist, axis = 1)          # assigns each sample to its closet centroid
           
            # updating
            updated_centroids = update_centroids(X,labels,self.K, old_centroids= self.centroids)# find the distance between the updated centroids and its prevous position
            convergence = np.linalg.norm(updated_centroids - self.centroids)                    # check if change in clusters flattens out
            self.centroids = updated_centroids                                                  # update stored centroids
            self.labels = labels

            if convergence <= self.tol: #break loop, clusters arent moving
                break

        # store final label, min distances, and the inertia         
        dist = compute_distance(X, self.centroids) # recompute distance for final centroids        
        min_dists = dist[np.arange(n_samples), self.labels]
        self.inertia = compute_inertia(min_dists)
           
        return self 



