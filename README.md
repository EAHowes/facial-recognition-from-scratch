# Facial Recognition from Scratch

**Team JME** — Jose Ortega · Mateo Bedoya · Ethan Howes

Comparing K-Means and DBSCAN clustering algorithms on synthetic facial embeddings, implemented from scratch in Python.

> **Note:** Individual file documentation is in [`/docs`](./docs). Navigate there for instructions on how to run each component.

---

## Overview

To focus on clustering algorithm development, we generate synthetic facial embeddings rather than processing real images. In theory, given embeddings pulled from real facial detection software, our from-scratch clustering algorithms should yield similar results.

---

## Project Structure

```text
facial-recognition-from-scratch
├── data
│   ├── dbscan_optimization_info.md
│   └── embeddings.py
├── docs
│   ├── data_generator_readme.md
│   ├── dbscan_README.md
│   └── k_means_readme.md
├── src
│   ├── dbscan.py
│   ├── dbscan_wrapper.py
│   ├── k_means.py
│   ├── k_means_wrapper.py
│   └── main.py
└── README.md
```

---

## Pipeline

Steps 1–3 are out of scope; our project picks up at the embedding stage.

1. ~~Face Detection — find where faces are in each image~~
2. ~~Face Alignment — normalize rotation, scale, and crop faces~~
3. ~~Feature Extraction / Embedding — convert each face into a numerical vector~~
4. **Embeddings** — generate synthetic facial embeddings for clustering
5. **Clustering** — group similar vectors together *(from scratch)*
6. **Labeling** — assign names or IDs to clusters once identity is known

---

## Getting Started

```bash
# 1. Generate embeddings
cd data
python embeddings.py

# 2. Run the CLI
cd ../src
python main.py
```

The CLI lets you run K-Means, DBSCAN, or both, and prints a side-by-side comparison of accuracy, runtime, and cluster statistics.

---

## Team

| Name | File | Role |
|------|------|------|
| Ethan Howes | `src/dbscan.py` | DBSCAN implementation |
| Mateo Bedoya | `src/k_means.py` | K-Means implementation |
| Jose Ortega | `data/embeddings.py` | Embedding generation |

---

## References

- [Segmentation of Brain Tumour from MRI image — Analysis of K-means and DBSCAN Clustering](https://www.semanticscholar.org/paper/Segmentation-of-Brain-Tumour-from-MRI-image-%E2%80%93-of-Bandyopadhyay/a082abca6c53cc8d4f5fc80c7ad0fa83464cca48) — Semantic Scholar
- [A Guide to the DBSCAN Clustering Algorithm](https://www.datacamp.com/tutorial/dbscan-clustering-algorithm) — DataCamp
- [DBSCAN Clustering in ML — Density Based Clustering](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/) — GeeksForGeeks
- [Create a K-Means Clustering Algorithm from Scratch in Python](https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670/) — Towards Data Science
- [VGGFace2](https://github.com/ox-vgg/vgg_face2) — labeled face library for same/different person validation
