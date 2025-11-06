```
This implementation of DBSCAN takes a while to run but it DOES WORK. 
Depending on your computer the script can take anywhere from 30 - 90 min to run. 
    - My laptop took about 80 minutes whereas my desktop took about 30 minutes on average.

Unfortunately due to the nature of numpy, the script is almost entirely CPU dependent and thus is rather slow.

The DBSCAN implementation and use is seperated across two files:

    dbscan.py                   O(n^2 x d)
        - Contains the core algorithm functionality for dbscan
        - Uses numpy for finding euclidean distance through matricies
        - Detects core points and grows clusters from them using region_query

    dbscan_wrapper.py           O(n^2 x d)
        - Loads embeddings
        - Assigns EPS and MINPTS
        - Calls dbscan and prints outputs to the console

-=-=-=-=-=-=-=-=-=-=-=-

HOW TO RUN DBSCAN:
    In a terminal run:
    - cd src/
    - python dbscan_wrapper.py

What you get:
    - Console output of eps
    - minpts
    - runtime
    - number of points
    - clusters found
    - number of noise points
    - purity/accuracy

Sample output:
    dbscan results
    eps: 1.007
    minPts: 10
    runtime: 2783.64 s
    points: 100000
    clusters: 10
    noise points: 28752
    Purity: 1.0000  (based on 71248 non-noise points)

-=-=-=-=-=-=-=-=-=-=-=-

```
