import numpy as np


def _dists(A, B):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


class LocalOutlierFactor:
    """Anomaly detection by comparing a point's LOCAL density with its neighbours' densities.

    "Far from everything" is the wrong test when clusters have different densities: a point 1 unit
    from a very tight cluster is a clear outlier, while 1 unit between points of a sparse cluster is
    normal. LOF asks "is this point sparser than the points around it?":

    1. k-distance(p): distance to p's k-th nearest neighbour.
    2. reach-dist(p, o) = max(k-distance(o), dist(p, o)) -- distances are floored by the neighbour's own scale.
    3. lrd(p) = 1 / mean reach-dist from p to its k neighbours   (local reachability density).
    4. LOF(p) = mean(lrd(neighbours)) / lrd(p).

    LOF ~ 1: as dense as its neighbours (inlier). LOF >> 1: much sparser than its neighbours (outlier).
    Scores and sign convention follow scikit-learn: `negative_outlier_factor_` = -LOF.

    ponytail: brute-force O(n^2) distance matrix; use a KD-tree / ball-tree for large n."""

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        k = self.n_neighbors
        D = _dists(X, X)
        np.fill_diagonal(D, np.inf)                       # a point is not its own neighbour
        self.neighbors_ = np.argsort(D, axis=1)[:, :k]
        nd = np.take_along_axis(D, self.neighbors_, axis=1)
        self.k_distance_ = nd[:, -1]
        reach = np.maximum(nd, self.k_distance_[self.neighbors_])
        self.lrd_ = 1.0 / np.maximum(reach.mean(axis=1), 1e-10)
        lof = self.lrd_[self.neighbors_].mean(axis=1) / self.lrd_
        self.negative_outlier_factor_ = -lof
        self.offset_ = np.quantile(-lof, self.contamination)  # below this -> outlier
        self.X_fit_ = X
        return self

    def score_samples(self, X):
        """-LOF of NEW points w.r.t. the training set (higher = more normal, ~ -1 for inliers)."""
        X = np.asarray(X, dtype=float)
        D = _dists(X, self.X_fit_)
        nbrs = np.argsort(D, axis=1)[:, : self.n_neighbors]
        nd = np.take_along_axis(D, nbrs, axis=1)
        reach = np.maximum(nd, self.k_distance_[nbrs])
        lrd = 1.0 / np.maximum(reach.mean(axis=1), 1e-10)
        return -(self.lrd_[nbrs].mean(axis=1) / lrd)

    def predict(self, X):
        """-1 outlier, 1 inlier, for new points."""
        return np.where(self.score_samples(X) < self.offset_, -1, 1)

    def fit_predict(self, X):
        """Labels for the training points themselves."""
        self.fit(X)
        return np.where(self.negative_outlier_factor_ < self.offset_, -1, 1)
