import numpy as np


class KMeans:
    """Lloyd's algorithm with k-means++ initialization."""

    def __init__(self, n_clusters=3, n_iters=300, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.centroids = self._kmeans_plusplus_init(X, rng)

        for _ in range(self.n_iters):
            labels = self._assign(X)
            new_centroids = np.array(
                [
                    X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                    for k in range(self.n_clusters)
                ]
            )
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = self._assign(X)
        self.inertia_ = self._inertia(X, self.labels_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign(X)

    def _assign(self, X):
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def _inertia(self, X, labels):
        return sum(
            np.sum((X[labels == k] - self.centroids[k]) ** 2) for k in range(self.n_clusters)
        )

    def _kmeans_plusplus_init(self, X, rng):
        n_samples = X.shape[0]
        centroids = [X[rng.integers(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dists / dists.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)
