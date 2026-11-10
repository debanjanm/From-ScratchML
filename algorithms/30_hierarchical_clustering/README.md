# Hierarchical (Agglomerative) Clustering

Build a tree of clusters from the bottom up: every point starts alone, then the **two closest clusters are merged**, over and over, until one cluster holds everything. The whole merge history is a *dendrogram*; cutting it at a height (or at `n_clusters` groups) gives a flat clustering. No need to choose `k` up front — and you can look at the tree to pick it.

"Closest" for two clusters depends on the **linkage**:

| Linkage | Distance between clusters | Tends to find |
|---|---|---|
| `single` | nearest pair of points | elongated / chained shapes (but sensitive to noise bridges) |
| `complete` | farthest pair of points | compact clusters |
| `average` | mean of all pairwise distances | a compromise |
| `ward` | smallest increase in within-cluster variance | round clusters of similar size (the usual default) |

After a merge, distances to the new cluster are computed from the old distances alone (Lance–Williams update), e.g. single: `min(d_ik, d_jk)`, average: `(n_i·d_ik + n_j·d_jk) / (n_i + n_j)`.

`linkage_matrix_` is in scipy's format — rows `[cluster_a, cluster_b, distance, size]`, merged clusters numbered `n, n+1, ...` — so `scipy.cluster.hierarchy.dendrogram` can plot it directly. The test and demo check the merge heights against `scipy.cluster.hierarchy.linkage`.

Cost is `O(n³)` time and `O(n²)` memory.

## Usage

```python
from model import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
model.labels_
model.linkage_matrix_    # the full tree
```

## Run

```bash
python test_model.py
python demo.py
```
