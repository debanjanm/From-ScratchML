import numpy as np


class AgglomerativeClustering:
    """Bottom-up hierarchical clustering. Start with every point as its own cluster and repeatedly
    merge the two CLOSEST clusters until one remains. The full merge history is kept in
    linkage_matrix_ (scipy's format), and cutting it at n_clusters gives labels_.

    How "closest" is defined for two clusters is the linkage:
        single    nearest pair of points        (can chain; finds elongated shapes)
        complete  farthest pair of points       (compact clusters)
        average   mean over all pairs           (a compromise)
        ward      smallest increase in total within-cluster variance (round, equal-sized clusters)

    Distances to a merged cluster are updated directly from the old ones (Lance-Williams
    formulas), so points are never revisited.

    ponytail: O(n^3) time, O(n^2) memory (full distance matrix, linear scan for the minimum).
    Fine for a few hundred points; use a nearest-neighbour-chain algorithm for more."""

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))
        if self.linkage == "ward":
            D = D**2  # ward's update is linear in SQUARED distances
        np.fill_diagonal(D, np.inf)

        ids = list(range(n))  # current cluster id sitting in each row (leaves are 0..n-1)
        size = np.ones(n)
        Z = []
        for step in range(n - 1):
            i, j = np.unravel_index(np.argmin(D), D.shape)
            i, j = min(i, j), max(i, j)
            d_ij = D[i, j]
            Z.append([min(ids[i], ids[j]), max(ids[i], ids[j]),
                      np.sqrt(d_ij) if self.linkage == "ward" else d_ij, size[i] + size[j]])

            d_i, d_j, n_i, n_j, n_k = D[i].copy(), D[j].copy(), size[i], size[j], size
            if self.linkage == "single":
                new = np.minimum(d_i, d_j)
            elif self.linkage == "complete":
                new = np.maximum(d_i, d_j)
            elif self.linkage == "average":
                new = (n_i * d_i + n_j * d_j) / (n_i + n_j)
            elif self.linkage == "ward":
                new = ((n_i + n_k) * d_i + (n_j + n_k) * d_j - n_k * d_ij) / (n_i + n_j + n_k)
            else:
                raise ValueError(f"unknown linkage: {self.linkage}")

            new[i] = new[j] = np.inf
            D[i, :] = new  # row/column i now stands for the merged cluster...
            D[:, i] = new
            D[j, :] = np.inf  # ...and j is retired
            D[:, j] = np.inf
            size[i] = n_i + n_j
            ids[i] = n + step

        self.linkage_matrix_ = np.array(Z)
        self.labels_ = self._cut(n, self.n_clusters)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _cut(self, n, n_clusters):
        """Replay the first n - n_clusters merges and read off the surviving clusters."""
        members = {i: [i] for i in range(n)}
        for step, (a, b, _, _) in enumerate(self.linkage_matrix_[: n - n_clusters]):
            members[n + step] = members.pop(int(a)) + members.pop(int(b))
        labels = np.empty(n, dtype=int)
        for label, leaves in enumerate(sorted(members.values(), key=min)):
            labels[leaves] = label
        return labels
