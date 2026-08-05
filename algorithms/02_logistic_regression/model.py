import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """Binary logistic regression trained by batch gradient descent on BCE loss."""

    def __init__(self, lr=0.1, n_iters=2000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = _sigmoid(z)
            error = y_pred - y
            grad_w = (1 / n_samples) * (X.T @ error)
            grad_b = (1 / n_samples) * np.sum(error)
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
