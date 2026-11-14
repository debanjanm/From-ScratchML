# Local Outlier Factor (LOF)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

"Far from everything" is the wrong test when clusters have **different densities**. A point 1 unit from a razor-tight cluster is clearly out of place; a point 1 unit from its neighbours in a sparse cluster is perfectly normal. LOF asks instead: **is this point sparser than the points around it?**

## How it works

For a point `p` and `k = n_neighbors`:

1. **k-distance(p)** — distance to `p`'s `k`-th nearest neighbour.
2. **Reachability distance** `reach(p, o) = max(k-distance(o), dist(p, o))` — distances are floored by the neighbour's own scale, which smooths out noise.
3. **Local reachability density** `lrd(p) = 1 / mean reach(p, o)` over `p`'s `k` neighbours.
4. **`LOF(p) = mean(lrd(o) for neighbours o) / lrd(p)`**

| LOF | Meaning |
|---|---|
| ≈ 1 | as dense as its neighbours — inlier |
| ≫ 1 | much sparser than its neighbours — outlier |

The sign follows scikit-learn: `negative_outlier_factor_ = −LOF`, and `contamination` sets the fraction flagged (`−1`) by taking that percentile of the training scores. The tests show a point that a plain "distance to the k-th neighbour" rule ranks as unremarkable while LOF puts it first, and the demo matches scikit-learn exactly for both training and new points.

`k` is the one knob: small is noisy, large smooths over small clusters of anomalies (a group of outliers bigger than `k` look like a normal cluster to each other).

Compare `39_isolation_forest`: that one needs no distances and scales better, but LOF is better when density varies across the data.

`ponytail:` brute-force `O(n²)` distance matrix; use a KD-/ball-tree for large `n`.

## Usage

```python
# (the class lives in notebook.ipynb)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X)            # -1 outlier, 1 inlier (training points)
lof.negative_outlier_factor_           # -LOF per training point
lof.predict(X_new)                     # new points, scored against the training set
```
