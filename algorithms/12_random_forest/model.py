import numpy as np


def _gini(y, n_classes):
    probs = np.bincount(y, minlength=n_classes) / len(y)
    return 1.0 - np.sum(probs**2)


class _Tree:
    """CART tree that only considers a random subset of features at each split."""

    def __init__(self, max_depth, min_samples_split, max_features, n_classes, rng):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_classes = n_classes
        self.rng = rng
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _leaf(self, y):
        return np.bincount(y, minlength=self.n_classes).argmax()

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return self._leaf(y)

        parent = _gini(y, self.n_classes)
        best_gain, best_feature, best_threshold = 0.0, None, None
        features = self.rng.choice(X.shape[1], self.max_features, replace=False)

        for f in features:
            values = np.unique(X[:, f])
            for t in (values[:-1] + values[1:]) / 2:  # midpoints between neighbours
                mask = X[:, f] <= t
                n_left = mask.sum()
                weighted = (
                    n_left * _gini(y[mask], self.n_classes)
                    + (len(y) - n_left) * _gini(y[~mask], self.n_classes)
                ) / len(y)
                if parent - weighted > best_gain:
                    best_gain, best_feature, best_threshold = parent - weighted, f, t

        if best_feature is None:
            return self._leaf(y)

        mask = X[:, best_feature] <= best_threshold
        return (
            best_feature,
            best_threshold,
            self._grow(X[mask], y[mask], depth + 1),
            self._grow(X[~mask], y[~mask], depth + 1),
        )

    def _traverse(self, x, node):
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if x[feature] <= threshold else right
        return node


class RandomForestClassifier:
    """Bagging of decision trees: bootstrap rows + random features per split, majority vote."""

    def __init__(self, n_estimators=25, max_depth=8, min_samples_split=2,
                 max_features="sqrt", random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.classes_ = None
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)  # any label type works
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        max_features = (
            max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else self.max_features
        )

        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n_samples, n_samples)  # bootstrap sample (with replacement)
            tree = _Tree(self.max_depth, self.min_samples_split, max_features, n_classes, rng)
            self.trees.append(tree.fit(X[idx], y_enc[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        n_classes = len(self.classes_)
        winners = [np.bincount(col, minlength=n_classes).argmax() for col in votes.T]
        return self.classes_[winners]
