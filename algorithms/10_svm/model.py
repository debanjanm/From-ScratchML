import numpy as np


class LinearSVM:
    """Binary linear SVM: minimizes  lam/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b)))
    by full-batch subgradient descent. Features should be on a similar scale."""

    def __init__(self, lam=0.01, lr=0.1, n_iters=1000):
        self.lam = lam
        self.lr = lr
        self.n_iters = n_iters
        self.classes_ = None
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("LinearSVM is binary: needs exactly 2 classes")
        y_pm = 2.0 * y_enc - 1.0  # {0, 1} -> {-1, +1}

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for t in range(self.n_iters):
            violators = y_pm * (X @ self.weights + self.bias) < 1  # inside margin or misclassified
            grad_w = self.lam * self.weights - (y_pm[violators] @ X[violators]) / len(y)
            grad_b = -y_pm[violators].sum() / len(y)
            step = self.lr / np.sqrt(t + 1)
            self.weights -= step * grad_w
            self.bias -= step * grad_b
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
