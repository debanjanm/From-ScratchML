import numpy as np


class DecisionTreeRegressor:
    """CART regression tree: each split minimizes squared error, each leaf predicts the mean."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X, dtype=float), np.asarray(y, dtype=float), depth=0)
        return self

    def predict(self, X):
        return self._predict(self.root, np.asarray(X, dtype=float))

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return y.mean()

        n, total = len(y), y.sum()
        best_gain, best_feature, best_threshold = 1e-12, None, None
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f])
            xs, ys = X[order, f], y[order]
            left_sum = np.cumsum(ys)[:-1]
            left_n = np.arange(1, n)
            # variance reduction, up to a constant: sum_l^2/n_l + sum_r^2/n_r - sum^2/n
            gain = left_sum**2 / left_n + (total - left_sum) ** 2 / (n - left_n) - total**2 / n
            gain = np.where(xs[1:] != xs[:-1], gain, -np.inf)  # can't split between equal values
            i = gain.argmax()
            if gain[i] > best_gain:
                best_gain, best_feature, best_threshold = gain[i], f, (xs[i] + xs[i + 1]) / 2

        if best_feature is None:
            return y.mean()

        mask = X[:, best_feature] <= best_threshold
        return (
            best_feature,
            best_threshold,
            self._grow(X[mask], y[mask], depth + 1),
            self._grow(X[~mask], y[~mask], depth + 1),
        )

    def _predict(self, node, X):
        if not isinstance(node, tuple):
            return np.full(len(X), node)
        feature, threshold, left, right = node
        mask = X[:, feature] <= threshold
        out = np.empty(len(X))
        out[mask] = self._predict(left, X[mask])
        out[~mask] = self._predict(right, X[~mask])
        return out
