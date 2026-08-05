import numpy as np


class LinearRegression:
    """Ordinary least squares regression: y = Xw + b."""

    def __init__(self, method="normal_equation", lr=0.05, n_iters=1000):
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.method == "normal_equation":
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias, self.weights = theta[0], theta[1:]
        elif self.method == "gradient_descent":
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            for _ in range(self.n_iters):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                grad_w = (2 / n_samples) * (X.T @ error)
                grad_b = (2 / n_samples) * np.sum(error)
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b
        else:
            raise ValueError(f"unknown method: {self.method}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias
