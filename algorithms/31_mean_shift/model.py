import numpy as np


def _distances(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def estimate_bandwidth(X, quantile=0.3):
    """Mean distance from each point to its ceil(quantile * n)-th nearest neighbour (itself included)."""
    X = np.asarray(X, dtype=float)
    k = max(1, int(len(X) * quantile))
    return np.sort(_distances(X, X), axis=1)[:, k - 1].mean()


class MeanShift:
    """Mode-seeking clustering. Imagine a hill of point density; every point climbs it:

        repeat:  move the point to the MEAN of all data points within `bandwidth` of it

    The mean always lies uphill of the current position (toward denser data), so every point
    ends up on a local density peak (a mode). Points that climb to the same peak form a cluster.
    The number of clusters is not given -- it falls out of the data and the bandwidth.

    bandwidth: radius of the window. Small -> many small clusters, large -> few big ones.
    Defaults to estimate_bandwidth(X). Flat (uniform) kernel.

    ponytail: every point is a seed and distances are recomputed each step, O(n^2) per iteration.
    Bin the seeds on a grid for large n."""

    def __init__(self, bandwidth=None, max_iter=300):
        self.bandwidth = bandwidth
        self.max_iter = max_iter

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        h = self.bandwidth if self.bandwidth is not None else estimate_bandwidth(X)
        self.bandwidth_ = h

        modes = []  # (center, number of points inside its final window)
        for seed in X:
            center = seed.copy()
            for _ in range(self.max_iter):
                inside = X[np.linalg.norm(X - center, axis=1) <= h]
                if len(inside) == 0:
                    break
                new_center = inside.mean(axis=0)
                converged = np.linalg.norm(new_center - center) < 1e-3 * h
                center = new_center
                if converged:
                    break
            modes.append((center, int((np.linalg.norm(X - center, axis=1) <= h).sum())))

        # many seeds reach (almost) the same peak: keep the strongest, drop anything within h of it
        modes.sort(key=lambda m: (-m[1], tuple(m[0])))
        kept = []
        for center, _ in modes:
            if all(np.linalg.norm(center - c) > h for c in kept):
                kept.append(center)
        self.cluster_centers_ = np.array(kept)
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        return _distances(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)
