import numpy as np


class _XGBTree:
    """One XGBoost tree. Every sample carries a gradient g and a hessian h of the loss. For a leaf
    holding a set of samples with sums G = sum(g), H = sum(h):

        best leaf weight   w = -G / (H + lambda)
        score of the leaf     G^2 / (H + lambda)        (higher = better fit)
        gain of a split       0.5 * (score_left + score_right - score_parent) - gamma

    lambda shrinks leaf weights (L2 regularization); gamma is the price of adding a leaf, so
    splits that don't improve the loss by more than gamma are never made (built-in pruning)."""

    def __init__(self, max_depth, reg_lambda, gamma, min_child_weight):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def fit(self, X, g, h):
        self.root = self._grow(X, g, h, 0)
        return self

    def predict(self, X):
        return self._predict(self.root, X)

    def _grow(self, X, g, h, depth):
        lam = self.reg_lambda
        G, H = g.sum(), h.sum()
        leaf = -G / (H + lam)
        if depth >= self.max_depth or len(g) < 2:
            return leaf

        best_gain, best_feature, best_threshold = 0.0, None, None
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f])
            xs = X[order, f]
            GL, HL = np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1]
            GR, HR = G - GL, H - HL
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)) - self.gamma
            valid = (xs[1:] != xs[:-1]) & (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            gain = np.where(valid, gain, -np.inf)
            i = gain.argmax()
            if gain[i] > best_gain:
                best_gain, best_feature, best_threshold = gain[i], f, (xs[i] + xs[i + 1]) / 2

        if best_feature is None:
            return leaf
        mask = X[:, best_feature] <= best_threshold
        return (best_feature, best_threshold,
                self._grow(X[mask], g[mask], h[mask], depth + 1),
                self._grow(X[~mask], g[~mask], h[~mask], depth + 1))

    def _predict(self, node, X):
        if not isinstance(node, tuple):
            return np.full(len(X), node)
        feature, threshold, left, right = node
        mask = X[:, feature] <= threshold
        out = np.empty(len(X))
        out[mask] = self._predict(left, X[mask])
        out[~mask] = self._predict(right, X[~mask])
        return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class XGBoost:
    """XGBoost-style gradient boosting: second-order (Newton) optimization with a regularized
    objective. Compared with plain gradient boosting it (1) uses the hessian as well as the
    gradient, (2) penalizes leaf weights (reg_lambda) and leaf count (gamma), and (3) prunes
    splits whose gain is not worth it.

    objective: "reg:squarederror" (g = pred - y, h = 1)
               "binary:logistic"  (g = p - y,    h = p(1 - p); labels any two values)

    ponytail: exact greedy split search only; no subsampling, column sampling or histogram
    binning, no missing-value handling, no multiclass."""

    def __init__(self, objective="reg:squarederror", n_estimators=100, learning_rate=0.3, max_depth=3,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.objective == "binary:logistic":
            self.classes_, y = np.unique(y, return_inverse=True)
            if len(self.classes_) != 2:
                raise ValueError("binary:logistic needs exactly 2 classes")
            p0 = y.mean()
            self.base_score = np.log(p0 / (1 - p0))  # prior as log-odds
        elif self.objective == "reg:squarederror":
            y = np.asarray(y, dtype=float)
            self.base_score = y.mean()
        else:
            raise ValueError(f"unknown objective: {self.objective}")

        F = np.full(len(y), self.base_score)
        self.trees = []
        for _ in range(self.n_estimators):
            if self.objective == "binary:logistic":
                p = _sigmoid(F)
                g, h = p - y, p * (1 - p)
            else:
                g, h = F - y, np.ones_like(y)
            tree = _XGBTree(self.max_depth, self.reg_lambda, self.gamma, self.min_child_weight)
            F += self.learning_rate * tree.fit(X, g, h).predict(X)
            self.trees.append(tree)
        return self

    def _raw(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.base_score)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        p = _sigmoid(self._raw(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        if self.objective == "binary:logistic":
            return self.classes_[(self._raw(X) > 0).astype(int)]
        return self._raw(X)
