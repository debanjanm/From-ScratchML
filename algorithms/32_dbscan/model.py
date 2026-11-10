import numpy as np


class DBSCAN:
    """Density-based clustering. labels_ holds cluster ids 0..k-1; noise points get -1.

    ponytail: full n x n distance matrix, O(n^2) memory. Use a KD-tree for large n."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        neighbors = [np.flatnonzero(row <= self.eps) for row in dist]  # includes the point itself
        is_core = np.array([len(nb) >= self.min_samples for nb in neighbors])

        labels = np.full(n, -1)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            labels[i] = cluster
            stack = [i]
            while stack:  # grow the cluster through chains of core points
                j = stack.pop()
                if not is_core[j]:
                    continue  # border points join the cluster but do not extend it
                for k in neighbors[j]:
                    if labels[k] == -1:
                        labels[k] = cluster
                        stack.append(k)
            cluster += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
