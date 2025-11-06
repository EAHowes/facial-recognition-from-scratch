Data generation:

How to run it:
- cd data
- python embeddings.py
- Takes like 3 seconds, creates embeddings.npy and labels.npy
- Dont commit the .npy files, its about 98mb

What you get:
- 100,000 samples
- 128 dimensions per sample
- 10 different identities (thats the k=10 if you need it)
- Each identity has exactly 10,000 samples
- Ground truth labels (0-9) so you can check accuracy

Loading the data:
- import numpy as np
- embeddings = np.load('data/embeddings.npy')
- labels = np.load('data/labels.npy')

About the data:
- All vectors normalized (length = 1.0)
- Data is already shuffled, identities aren't grouped together
- Same identity samples are similar but not identical, like different photos of same person

Testing tips:
- Test on small data first, embeddings[:1000] and labels[:1000]
- Once it works on 1k, scale to 100k
- Same seed (42) means we all get the same data

Typical distances you'll see:
- Same identity: around 0.7
- Different identity: around 1.0