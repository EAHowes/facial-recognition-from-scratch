```text
This project aims to find an optimal solution for facial recognition clustering by comparing kmeans and dbscan algorithms.

Tree:
facial-recognition-from-scratch
├── data
│   └── embeddings.py
├── src
│   ├── dbscan.py
│   ├── k-means.py
│   └── main.py
└── README.md

-=-=-=-

General Pipeline:
1. Face Detection → Find where faces are in each image                              # Not from scratch
2. Face Alignment → Normalize rotation, scale, and crop faces                       # Not from scratch
3. Feature Extraction / Embedding → Convert each face into a numerical vector       # Not from scratch

(Where our project scope begins)
4. Clustering → Group similar vectors together                                      # From scratch
5. Labeling → Assign names or IDs to clusters once you know who’s who               # Using clustering

Raw Images  →  Face Detection  →  Alignment  →  Embeddings.py  →  Clustering

-=-=-=-=-

Sources:

    Medium - DBSCAN Algorithm from Scratch in Python
        https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c

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
├── dbscan.py                       # Scratch implementation of dbscan ... somehow optimize for 100,000 faces
└── utils.py                        # Helpers that are shared for both implentations

    Open responsibilities:
        - Algorithms
            - k-means implementation
            - DBSCAN implementation
        - CLI / IO 
            - main.py
            - utils.py 
        - Embedding Gen / evaluation
            - embeddings.py                         # likely use a pretrained embedder
            - README.md                             # explain how to run code
        - Facial Detection / Alignment              # to make the project end-to-end (start from raw images)
            - detect_faces.py
            - align.py

    Ethan
        - None established
    Mateo
        - None established
    Jose
        - None established
```
