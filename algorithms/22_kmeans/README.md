# K-Means

Lloyd's algorithm: initialize centroids via k-means++ (spread out, weighted by squared distance to nearest existing centroid), then alternate assign-points-to-nearest-centroid and recompute-centroid-as-mean until centroids stop moving (or `n_iters` is hit).

## Usage

```python
from model import KMeans

model = KMeans(n_clusters=3).fit(X)
labels = model.labels_        # cluster assignment used during fit
new_labels = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
