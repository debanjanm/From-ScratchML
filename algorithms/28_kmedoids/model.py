import numpy as np


def _pairwise(A, B, metric):
    diff = A[:, None, :] - B[None, :, :]
    if metric == "euclidean":
        return np.sqrt((diff**2).sum(axis=2))
    if metric == "manhattan":
        return np.abs(diff).sum(axis=2)
    raise ValueError(f"unknown metric: {metric}")


class KMedoids:
    """Like K-Means, but each cluster is represented by a MEDOID: an actual data point, the
    member with the smallest total distance to the rest of its cluster. Consequences:

    - robust to outliers (a median-like centre, not a mean that gets dragged),
    - works with any distance metric (the algorithm only reads a distance matrix),
    - cluster centres are always real, interpretable samples.

    Alternates assign -> re-pick each medoid (Voronoi iteration) until nothing changes, restarted
    n_init times from different seeds; the lowest-cost run wins.

    ponytail: Voronoi iteration, not the full PAM swap search; it can settle in a slightly worse
    local optimum. O(n^2) memory for the distance matrix."""

    def __init__(self, n_clusters=3, metric="euclidean", max_iter=100, n_init=5, random_state=0):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, k = len(X), self.n_clusters
        D = _pairwise(X, X, self.metric)
        rng = np.random.default_rng(self.random_state)

        best_cost = np.inf
        for _ in range(self.n_init):
            medoids = self._init(D, rng)
            for _ in range(self.max_iter):
                labels = D[:, medoids].argmin(axis=1)
                new = medoids.copy()
                for c in range(k):
                    members = np.flatnonzero(labels == c)
                    if len(members):  # the member closest to all the others becomes the medoid
                        new[c] = members[D[np.ix_(members, members)].sum(axis=1).argmin()]
                if np.array_equal(new, medoids):
                    break
                medoids = new
            labels = D[:, medoids].argmin(axis=1)
            cost = D[np.arange(n), medoids[labels]].sum()
            if cost < best_cost:
                best_cost, best_medoids, best_labels = cost, medoids, labels

        self.medoid_indices_ = best_medoids
        self.cluster_centers_ = X[best_medoids]
        self.labels_ = best_labels
        self.inertia_ = best_cost  # total distance from every point to its medoid
        return self

    def _init(self, D, rng):
        # k-means++ style: spread the starting medoids out, far points more likely
        medoids = [rng.integers(len(D))]
        for _ in range(1, self.n_clusters):
            nearest = D[:, medoids].min(axis=1) ** 2
            medoids.append(rng.choice(len(D), p=nearest / nearest.sum()))
        return np.array(medoids)

    def predict(self, X):
        return _pairwise(np.asarray(X, dtype=float), self.cluster_centers_, self.metric).argmin(axis=1)
