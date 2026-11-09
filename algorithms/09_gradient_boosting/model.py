import numpy as np


class _RegressionTree:
    """Small CART regression tree: splits minimize squared error, leaves predict the mean."""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        return self._predict(self.root, X)

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
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


class GradientBoostingRegressor:
    """Squared-error gradient boosting: each tree fits the residuals of the ensemble so far."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.init_ = None
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_ = y.mean()
        pred = np.full(len(y), self.init_)
        self.trees = []
        for _ in range(self.n_estimators):
            residual = y - pred  # negative gradient of 1/2 * squared error
            tree = _RegressionTree(self.max_depth, self.min_samples_split).fit(X, residual)
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
