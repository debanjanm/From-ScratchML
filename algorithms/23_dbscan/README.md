# DBSCAN

Density-based clustering. No `k` to choose, finds arbitrarily shaped clusters, and labels outliers as noise (`-1`).

- A point is a **core point** if at least `min_samples` points (itself included) lie within distance `eps`.
- A cluster starts at an unvisited core point and grows through every core point reachable from it. **Border points** (within `eps` of a core point, but not core themselves) join the cluster without extending it.
- Anything left over is **noise**.

Compared to K-Means (`22_kmeans`): handles non-convex shapes such as two moons, and needs no cluster count — but `eps` must suit the data's scale.

Uses a full pairwise distance matrix (O(n²) memory), fine for the sizes this repo targets.

## Usage

```python
from model import DBSCAN

labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)   # -1 = noise
```

## Run

```bash
python test_model.py
python demo.py
```
