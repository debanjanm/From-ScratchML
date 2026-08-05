from collections import Counter

import numpy as np


class KNNClassifier:
    """Brute-force k-nearest-neighbors classifier (Euclidean distance, majority vote)."""

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        dists = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(dists)[: self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]
