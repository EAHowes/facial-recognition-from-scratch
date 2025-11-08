```
The K-Means w/ Cosine distance implementation and use is seperated across two files:

    k_means.py                   O(n × k × d × i)
       n = n-samples 
       k = number of clusters
       d = dimensions of each data point
       i = iterations until convergence

    k_means_wrapper.py           O(n × k × d × i)
       n = n-samples 
       k = number of clusters
       d = dimensions of each data point
       i = iterations until convergence

-=-=-=-=-=-=-=-=-=-=-=-

HOW TO RUN K-MEANS:
    In a terminal run:
    - cd src/
    - python -m src.k_means_wrapper

What you get:
    - Number of iterations - How many iterations are actually preformed 
    - runtime              - Excution time
    - centroids shape      - (number of clusters , dimensions)
    - Inertia              - Clustering compactness measure (sum of minimum cosine distances)
    - Cluster counts       - Number of samples assigned to each cluster after convergence
    - First 10 labels      - Labels from 0-9 (K = 10)
    

Sample output:

 Running K-Means with K=10
--- K-Means Results ---
    Number of iterations 3
    runtime: 0.15 seconds
    centroids shape: (10, 128)
    Inertia: 4930.976106084883
    Cluster counts: [1000 1000 1000 1000 1000 1000 1000 1000 1000 1000]
    First 10 labels: [2 3 5 2 0 1 7 1 3 9]
```
