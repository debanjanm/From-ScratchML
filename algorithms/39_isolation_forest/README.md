# Isolation Forest

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Most anomaly detectors first model what "normal" looks like, then flag what doesn't fit. Isolation Forest flips it: **anomalies are few and different, so they are easy to isolate.**

## How it works

1. Build many random trees. Each split picks a **random feature** and a **random cut point** between that feature's min and max — no labels, no distances, no gain criterion.
2. A point in a dense cluster is surrounded by neighbours, so it takes many cuts to get it alone in a leaf (**long path**). A far-off point is alone after a cut or two (**short path**).
3. Average the path length `E[h(x)]` over all trees (a leaf that still holds `m` points is credited `c(m)`, the average depth needed to separate them) and normalize:

   `score(x) = 2^(−E[h(x)] / c(ψ))`

   Near **1** = anomaly, around **0.5** = nothing stands out, well below 0.5 = deep in the crowd.
4. `predict` flags the top `contamination` fraction of training scores as `−1`.

Each tree sees only a random subsample of `ψ = 256` points and is cut off at depth `⌈log₂ ψ⌉`. That keeps it fast (near linear in `n`) and stops dense clusters from hiding anomalies — more data is not better here.

Good for tabular data with numeric features and no labels. It can miss anomalies that sit *inside* a cluster's range, and axis-aligned random cuts struggle when anomalies are only visible as a combination of features. See `40_local_outlier_factor` for a density-based take.

Note: scikit-learn's `score_samples` returns the **negative** of this score (higher = more normal). The demo flips it to compare.

`ponytail:` nested tuples and Python recursion — fine for thousands of points; use flat arrays or sklearn for millions.

## Usage

```python
# (the class lives in notebook.ipynb)

forest = IsolationForest(n_estimators=100, contamination=0.05).fit(X)
forest.score_samples(X)    # in (0, 1), higher = more anomalous
forest.predict(X_new)      # -1 anomaly, 1 normal
```
