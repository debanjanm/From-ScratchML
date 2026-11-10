import numpy as np


class _ExtraTree:
    """At each node: pick random features, pick a RANDOM threshold for each (uniform between that
    feature's min and max in the node), and keep whichever random split has the best Gini score.
    No searching for the optimal threshold -- that is the whole idea."""

    def __init__(self, max_depth, min_samples_split, max_features, n_classes, rng):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_classes = n_classes
        self.rng = rng

    def fit(self, X, y):
        self.root = self._grow(X, np.eye(self.n_classes)[y], 0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _grow(self, X, Y, depth):
        counts = Y.sum(axis=0)
        n = len(Y)
        if depth >= self.max_depth or n < self.min_samples_split or counts.max() == n:
            return counts.argmax()

        best_score, best_feature, best_threshold = -np.inf, None, None
        for f in self.rng.choice(X.shape[1], self.max_features, replace=False):
            lo, hi = X[:, f].min(), X[:, f].max()
            if lo == hi:
                continue
            t = self.rng.uniform(lo, hi)  # random cut point
            mask = X[:, f] <= t
            left, right = Y[mask].sum(axis=0), Y[~mask].sum(axis=0)
            # sum(c^2/n) over both sides; larger == lower weighted Gini impurity
            score = (left**2).sum() / mask.sum() + (right**2).sum() / (~mask).sum()
            if score > best_score:
                best_score, best_feature, best_threshold = score, f, t

        if best_feature is None:
            return counts.argmax()
        mask = X[:, best_feature] <= best_threshold
        return (best_feature, best_threshold,
                self._grow(X[mask], Y[mask], depth + 1), self._grow(X[~mask], Y[~mask], depth + 1))

    def _traverse(self, x, node):
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if x[feature] <= threshold else right
        return node


class ExtraTreesClassifier:
    """Extremely Randomized Trees. Like Random Forest, but split thresholds are drawn at random
    instead of optimized, and by default every tree sees the WHOLE training set (no bootstrap).

    More randomness -> the trees are more decorrelated and each one is cheaper to build (no sorting
    or scanning for the best cut), so the ensemble has lower variance, at the price of slightly
    higher bias per tree."""

    def __init__(self, n_estimators=50, max_depth=12, min_samples_split=2, max_features="sqrt",
                 bootstrap=False, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        max_features = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else self.max_features

        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = _ExtraTree(self.max_depth, self.min_samples_split, max_features, k, rng)
            self.trees.append(tree.fit(X[idx], y_enc[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.array([t.predict(X) for t in self.trees])
        k = len(self.classes_)
        return self.classes_[[np.bincount(col, minlength=k).argmax() for col in votes.T]]
