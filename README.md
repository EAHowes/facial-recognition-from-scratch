```text
INDIVIDUAL FILE READMEs ARE LOCATED IN /docs ... NAVIGATE THERE FOR INSTRUCTIONS ON HOW TO RUN FILES

This project aims to find an optimal solution for facial recognition clustering by comparing kmeans and dbscan algorithms.

To focus on clustering algoritm development, we chose to create synthetic facial embeddings for testing.
In theory, given embeddings pulled from real facial detection softwares, our from-scratch clustering algos should yield similar results.

-=-=-=-

Tree:
facial-recognition-from-scratch
├── data
│   ├── dbscan_optimization_info.md
│   └── embeddings.py
├── docs
│   ├── data_generator_readme.md
│   ├── dbscan_README.md
│   └── k_means_readme.md
├── src
│   ├── dbscan.py
│   ├── dbscan_wrapper.py
│   ├── k_means.py
│   ├── k_means_wrapper.py
│   └── main.py
└── README.md

-=-=-=-

General Pipeline:
1. Face Detection → Find where faces are in each image                              # Not in scope 
2. Face Alignment → Normalize rotation, scale, and crop faces                       # Not in scope
3. Feature Extraction / Embedding → Convert each face into a numerical vector       # Not in scope
   
(Where our project scope begins)
3. Embeddings → Create synthetic facial embeddings to cluster
4. Clustering → Group similar vectors together                                      # From scratch
5. Labeling → Assign names or IDs to clusters once you know who’s who               # Using clustering

-=-=-=-=-

Sources:

    Semantic Scholar - Segmentation of Brain Tumour from MRI image – Analysis of K-means and DBSCAN Clustering
        https://www.semanticscholar.org/paper/Segmentation-of-Brain-Tumour-from-MRI-image-%E2%80%93-of-Bandyopadhyay/a082abca6c53cc8d4f5fc80c7ad0fa83464cca48

    Data Camp - A Guide to the DBSCAN Clustering Algorithm
        https://www.datacamp.com/tutorial/dbscan-clustering-algorithm

    Geeks for Geeks - DBSCAN Clustering in ML - Density based Clustering
        https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/

    Towards Data Science - Create a K-Means Clustering Algorithm from Scratch in Python
        https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670/

    VGGFace2 
        Library of faces with labels for same / different person

-=-=--=-

Responsibilities: 

sample files (not a tree)
├── main.py                         # Puts everything together and runs kmeans/dbscan on embeddings.npy
├── embeddings.py                   # Converts images to vectors and saves them into .npy files for clustering 
├── k-means.py                      # Scratch implementation of k-means ... uses cosine distance by default
└── dbscan.py                       # Scratch implementation of dbscan ... somehow optimize for 100,000 faces

    Open responsibilities:
        - Algorithms
            - k-means implementation
            - DBSCAN implementation
        - CLI / IO 
            - main.py
        - Embedding Gen 
            - embeddings.py                         
            - README.md                     # explain how to run code
        - Facial Detection / Alignment      # Implement if we have extra time ... to make the project end-to-end (start from raw images)
            - detect_faces.py
            - align.py

    Ethan
        - dbscan.py
    Mateo
        - k-means.py
    Jose
        - embeddings.py

```
