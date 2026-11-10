import numpy as np


def _kmeans(X, k, rng, n_iters=100):
    """Tiny k-means++ / Lloyd, used on the spectral embedding."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
    centers = np.array(centers)
    for _ in range(n_iters):
        labels = ((X[:, None] - centers[None]) ** 2).sum(-1).argmin(axis=1)
        new = np.array([X[labels == c].mean(axis=0) if (labels == c).any() else centers[c] for c in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels


class SpectralClustering:
    """Cluster by CONNECTIVITY instead of by distance to a centre, using a graph.

    1. Build an affinity matrix W: how strongly each pair of points is connected
       ("rbf": exp(-gamma * dist^2) for all pairs, or "nearest_neighbors": linked to their k nearest).
    2. Form the normalized matrix  M = D^-1/2 W D^-1/2  (D = diagonal of row sums). Its top-k
       eigenvectors are the bottom-k eigenvectors of the graph Laplacian  L = I - M.
    3. Each point becomes a row of that eigenvector matrix (its "spectral embedding"). In this new
       space, points in the same connected blob sit together -- even if the blob was a curve.
    4. Row-normalize and run plain K-Means in the embedding.

    This is how it separates shapes like two interleaved moons that K-Means (which assumes round
    clusters) cannot.

    ponytail: dense O(n^2) affinity and a full eigendecomposition O(n^3); use sparse graphs and
    Lanczos for large n."""

    def __init__(self, n_clusters=2, affinity="nearest_neighbors", gamma=1.0, n_neighbors=10,
                 random_state=0):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _affinity_matrix(self, X):
        n = len(X)
        d2 = ((X[:, None] - X[None]) ** 2).sum(axis=2)
        if self.affinity == "rbf":
            W = np.exp(-self.gamma * d2)
        elif self.affinity == "nearest_neighbors":
            neighbors = np.argsort(d2, axis=1)[:, 1:self.n_neighbors + 1]  # skip the point itself
            A = np.zeros((n, n))
            A[np.repeat(np.arange(n), self.n_neighbors), neighbors.ravel()] = 1.0
            W = 0.5 * (A + A.T)  # symmetric: linked if either is among the other's neighbours
        else:
            raise ValueError(f"unknown affinity: {self.affinity}")
        np.fill_diagonal(W, 0.0)
        return W

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        W = self._affinity_matrix(X)
        d_inv_sqrt = 1.0 / np.sqrt(W.sum(axis=1))
        M = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]

        eigenvalues, eigenvectors = np.linalg.eigh(M)  # ascending order
        embedding = eigenvectors[:, -self.n_clusters:]  # top-k = smallest Laplacian eigenvalues
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        self.affinity_matrix_ = W
        self.embedding_ = embedding
        self.labels_ = _kmeans(embedding, self.n_clusters, np.random.default_rng(self.random_state))
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
