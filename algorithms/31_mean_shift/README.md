# Mean Shift

A clustering algorithm that **finds the number of clusters by itself**. Picture the data as a landscape where dense regions are hills. Put a window of radius `bandwidth` on every point and repeatedly:

> move the point to the **mean** of all data points inside its window

That mean is always uphill (toward where the data is denser), so every point climbs to the nearest **mode** — a local density peak. Points that reach the same peak share a cluster. Seeds that converge to nearly the same place (within one `bandwidth`) are merged, keeping the one with the most points.

| | K-Means | Mean Shift |
|---|---|---|
| Need `k`? | yes | **no** — the single parameter is `bandwidth` |
| Cluster shape | round | any shape the density has |
| Outliers | pull centres | become their own tiny mode (or absorbed) |
| Cost | cheap | `O(n²)` per iteration |

`bandwidth` is the only knob: small gives many fragments, large gives few merged blobs. If omitted it is estimated as the average distance to the `quantile`-th nearest neighbour (`estimate_bandwidth`, same rule as scikit-learn). This implementation uses a flat kernel, like scikit-learn's, so the results line up.

## Usage

```python
from model import MeanShift

model = MeanShift(bandwidth=2.0).fit(X)
model.cluster_centers_    # the modes it discovered
model.labels_
model.predict(X_new)      # nearest mode
```

## Run

```bash
python test_model.py
python demo.py
```
