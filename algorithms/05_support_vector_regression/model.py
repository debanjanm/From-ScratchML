import numpy as np


class LinearSVR:
    """Linear support vector regression: fit a line with a tube of half-width `epsilon` around it.
    Errors inside the tube cost nothing; outside, they cost linearly. Minimizes

        lam/2 * ||w||^2 + mean(max(0, |y - (Xw + b)| - epsilon))

    by full-batch subgradient descent. Scale features (and y, since epsilon is in y's units).

    ponytail: linear kernel only, no SMO / dual solver. Add kernels when needed."""

    def __init__(self, epsilon=0.1, lam=0.001, lr=0.1, n_iters=2000):
        self.epsilon = epsilon
        self.lam = lam
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self.weights = np.zeros(X.shape[1])
        self.bias = y.mean()  # start at the mean: the tube only cares about the shape from there

        for t in range(self.n_iters):
            residual = y - (X @ self.weights + self.bias)
            direction = np.sign(residual) * (np.abs(residual) > self.epsilon)  # 0 inside the tube
            grad_w = self.lam * self.weights - (X.T @ direction) / n
            grad_b = -direction.sum() / n
            step = self.lr / np.sqrt(t + 1)
            self.weights -= step * grad_w
            self.bias -= step * grad_b
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias
