import numpy as np


def _softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class _RegressionTree:
    """Regression tree fit to residuals. Splits are chosen by variance reduction (like
    07_decision_tree_regressor), but each leaf outputs a Newton step  scale * sum(r) / sum(h),
    which accounts for the curvature h of the log loss instead of just averaging residuals."""

    def __init__(self, max_depth, min_samples_split, scale):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.scale = scale

    def fit(self, X, r, h):
        self.root = self._grow(X, r, h, 0)
        return self

    def predict(self, X):
        return self._predict(self.root, X)

    def _leaf(self, r, h):
        denom = h.sum()
        return 0.0 if denom < 1e-12 else self.scale * r.sum() / denom

    def _grow(self, X, r, h, depth):
        n = len(r)
        if depth >= self.max_depth or n < self.min_samples_split:
            return self._leaf(r, h)

        total = r.sum()
        best_gain, best_feature, best_threshold = 1e-12, None, None
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f])
            xs, rs = X[order, f], r[order]
            left_sum, left_n = np.cumsum(rs)[:-1], np.arange(1, n)
            gain = left_sum**2 / left_n + (total - left_sum) ** 2 / (n - left_n) - total**2 / n
            gain = np.where(xs[1:] != xs[:-1], gain, -np.inf)
            i = gain.argmax()
            if gain[i] > best_gain:
                best_gain, best_feature, best_threshold = gain[i], f, (xs[i] + xs[i + 1]) / 2

        if best_feature is None:
            return self._leaf(r, h)
        mask = X[:, best_feature] <= best_threshold
        return (best_feature, best_threshold,
                self._grow(X[mask], r[mask], h[mask], depth + 1),
                self._grow(X[~mask], r[~mask], h[~mask], depth + 1))

    def _predict(self, node, X):
        if not isinstance(node, tuple):
            return np.full(len(X), node)
        feature, threshold, left, right = node
        mask = X[:, feature] <= threshold
        out = np.empty(len(X))
        out[mask] = self._predict(left, X[mask])
        out[~mask] = self._predict(right, X[~mask])
        return out


class GradientBoostingClassifier:
    """Gradient boosting for classification with any number of classes (multinomial deviance).

    The model keeps one raw score F_k(x) per class; probabilities are softmax(F). Each round,
    for every class k, fit a small tree to the gradient of the log loss -- the residual
    (1[y = k] - p_k) -- and add it to F_k, scaled by the learning rate. Compare 22_gradient_boosting,
    which does the same with squared error and a single score."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        n, k = len(X), len(self.classes_)
        Y = np.eye(k)[y_enc]

        self.init_ = np.log(Y.mean(axis=0))  # start from the class priors (as log-odds)
        F = np.tile(self.init_, (n, 1))
        self.trees, self.loss_history_ = [], []
        for _ in range(self.n_estimators):
            P = _softmax(F)
            self.loss_history_.append(-np.mean(np.log(P[np.arange(n), y_enc] + 1e-12)))
            round_trees = []
            for c in range(k):
                residual = Y[:, c] - P[:, c]   # negative gradient of the log loss
                hessian = P[:, c] * (1 - P[:, c])
                tree = _RegressionTree(self.max_depth, self.min_samples_split, (k - 1) / k)
                F[:, c] += self.learning_rate * tree.fit(X, residual, hessian).predict(X)
                round_trees.append(tree)
            self.trees.append(round_trees)
        return self

    def _raw_scores(self, X):
        X = np.asarray(X, dtype=float)
        F = np.tile(self.init_, (len(X), 1))
        for round_trees in self.trees:
            for c, tree in enumerate(round_trees):
                F[:, c] += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        return _softmax(self._raw_scores(X))

    def predict(self, X):
        return self.classes_[self._raw_scores(X).argmax(axis=1)]
