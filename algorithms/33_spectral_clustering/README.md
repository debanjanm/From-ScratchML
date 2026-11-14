# Spectral Clustering

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

K-Means (`27_kmeans`) assumes round clusters around a centre, so it cuts two interleaved moons or two concentric rings straight through. Spectral clustering groups points by **connectivity** instead: points that are linked by a chain of near neighbours belong together, whatever the shape.

## How it works

1. **Affinity graph.** Build a matrix `W` of how strongly each pair of points is connected:
   - `nearest_neighbors`: link each point to its `n_neighbors` closest (symmetrized) — the usual choice for curved shapes
   - `rbf`: `exp(−γ‖x − x'‖²)` for every pair
2. **Normalize.** With `D` the diagonal of row sums, take `M = D^(−1/2) · W · D^(−1/2)`. Its top-`k` eigenvectors are the bottom-`k` eigenvectors of the graph Laplacian `L = I − M`. A graph with `k` well-separated blobs has exactly `k` eigenvalues near 0 whose eigenvectors are (roughly) indicators of the blobs.
3. **Embed.** Each point becomes one row of that eigenvector matrix. In this space the tangled shapes are pulled apart into compact groups.
4. **Cluster the embedding** with ordinary K-Means (a small one is built in, so the folder stays self-contained).

It is a relaxed, tractable version of the graph-cut problem: cut the graph into `k` pieces while severing as little connection as possible.

Cost: dense `O(n²)` memory and an `O(n³)` eigendecomposition — fine for a few hundred to a few thousand points. You choose `k`; the demo and test show K-Means failing on the same data.

## Usage

```python
# (the class lives in notebook.ipynb)

labels = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=10).fit_predict(X)
labels = SpectralClustering(n_clusters=2, affinity="rbf", gamma=20.0).fit_predict(X)
```
