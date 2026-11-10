# K-Medoids

K-Means (`27_kmeans`) with one change: a cluster is represented by a **medoid** — an actual data point, the member with the smallest total distance to all the others — instead of a mean.

| | K-Means | K-Medoids |
|---|---|---|
| Centre | mean (may not be a real sample) | medoid (always a real sample) |
| Outliers | drag the mean | barely move the medoid |
| Distance | squared Euclidean | any metric (Euclidean, Manhattan, ...) |
| Cost | cheap | `O(n²)` — needs the pairwise distance matrix |

The algorithm alternates two steps until nothing changes: assign every point to its nearest medoid; then in each cluster, pick the member minimizing the summed distance to its cluster-mates as the new medoid. It is restarted from several k-means++-style seeds and the lowest-cost result is kept.

This is the Voronoi-iteration variant, which is simpler and faster than the classic PAM swap search but can land in a slightly worse optimum.

## Usage

```python
from model import KMedoids

model = KMedoids(n_clusters=3, metric="manhattan").fit(X)
model.labels_            # cluster of each training point
model.cluster_centers_   # the medoids (rows of X)
model.predict(X_new)
```

## Run

```bash
python test_model.py
python demo.py
```
