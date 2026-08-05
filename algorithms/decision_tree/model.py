from collections import Counter

import numpy as np


class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # set only on leaf nodes

    def is_leaf(self):
        return self.value is not None


def _gini(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1.0 - np.sum(probs**2)


class DecisionTreeClassifier:
    """CART decision tree for classification, splitting on Gini impurity."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])

    def _grow(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return _Node(value=Counter(y).most_common(1)[0][0])

        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            return _Node(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, best_feature] <= best_threshold
        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_gain = 0.0
        best_feature, best_threshold = None, None
        parent_gini = _gini(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if left_mask.sum() == 0 or (~left_mask).sum() == 0:
                    continue
                n = len(y)
                gini_left = _gini(y[left_mask])
                gini_right = _gini(y[~left_mask])
                weighted = (left_mask.sum() / n) * gini_left + ((~left_mask).sum() / n) * gini_right
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
