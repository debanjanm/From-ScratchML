# K-Means

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Lloyd's algorithm: initialize centroids via k-means++ (spread out, weighted by squared distance to nearest existing centroid), then alternate assign-points-to-nearest-centroid and recompute-centroid-as-mean until centroids stop moving (or `n_iters` is hit).

## Usage

```python
# (the class lives in notebook.ipynb)

model = KMeans(n_clusters=3).fit(X)
labels = model.labels_        # cluster assignment used during fit
new_labels = model.predict(X_test)
```
