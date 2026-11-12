import numpy as np


def _c(n):
    """Average path length of an unsuccessful search in a binary search tree of n points.
    Used to normalize path lengths and to credit leaves that still hold several points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _grow(X, depth, limit, rng):
    """Random tree: pick a random feature, split at a random value between its min and max.
    Returns an int (leaf holding that many points) or (feature, split, left, right)."""
    n = len(X)
    if depth >= limit or n <= 1:
        return n
    features = np.flatnonzero(X.max(axis=0) > X.min(axis=0))
    if len(features) == 0:  # all points identical
        return n
    f = rng.choice(features)
    split = rng.uniform(X[:, f].min(), X[:, f].max())
    left = X[:, f] < split
    return (f, split, _grow(X[left], depth + 1, limit, rng), _grow(X[~left], depth + 1, limit, rng))


def _path_lengths(node, X, idx, depth, out):
    if not isinstance(node, tuple):
        out[idx] = depth + _c(node)
        return
    f, split, left, right = node
    go_left = X[idx, f] < split
    _path_lengths(left, X, idx[go_left], depth + 1, out)
    _path_lengths(right, X, idx[~go_left], depth + 1, out)


class IsolationForest:
    """Anomaly detection by how EASY a point is to isolate.

    Build many random trees: each split picks a random feature and a random cut point. Anomalies are
    few and different, so a handful of random cuts already puts them alone in a leaf -- a SHORT path
    from the root. Normal points sit in dense regions and need many cuts -- a LONG path.

        score(x) = 2 ** (-E[path length of x] / c(psi))        in (0, 1)

    near 1 = anomaly, around 0.5 = nothing stands out, well below 0.5 = deep inside the crowd.
    Each tree is grown on a small random subsample (`max_samples`, default 256), which both makes it
    fast and stops dense clusters from masking the anomalies.

    Note: scikit-learn's `score_samples` returns the NEGATIVE of this score (higher = more normal).

    ponytail: nodes are nested tuples and recursion is plain Python, fine for thousands of points;
    use flat arrays (or sklearn) for big data."""

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1, random_state=0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.psi_ = min(self.max_samples, len(X))
        limit = int(np.ceil(np.log2(max(self.psi_, 2))))  # stop at ~average tree height; anomalies are shallow
        self.trees_ = [
            _grow(X[rng.choice(len(X), self.psi_, replace=False)], 0, limit, rng)
            for _ in range(self.n_estimators)
        ]
        self.threshold_ = np.quantile(self.score_samples(X), 1 - self.contamination)
        return self

    def score_samples(self, X):
        """Anomaly score in (0, 1); higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        total = np.zeros(len(X))
        for tree in self.trees_:
            lengths = np.zeros(len(X))
            _path_lengths(tree, X, np.arange(len(X)), 0, lengths)
            total += lengths
        return 2.0 ** (-(total / self.n_estimators) / _c(self.psi_))

    def predict(self, X):
        """-1 for anomalies (top `contamination` fraction of training scores), 1 for normal."""
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def fit_predict(self, X):
        return self.fit(X).predict(X)
