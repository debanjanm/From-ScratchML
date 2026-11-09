import numpy as np


def _center(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X, y, X.mean(axis=0), y.mean()


class RidgeRegression:
    """Least squares + L2 penalty:  ||y - Xw||^2 + alpha * ||w||^2   (closed form).
    The bias is not penalized: data is centered, then the bias is recovered from the means."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X, y, x_mean, y_mean = _center(X, y)
        Xc, yc = X - x_mean, y - y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.weights = np.linalg.solve(A, Xc.T @ yc)
        self.bias = y_mean - x_mean @ self.weights
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias


class LassoRegression:
    """Least squares + L1 penalty:  1/(2n) * ||y - Xw||^2 + alpha * ||w||_1
    solved by cyclic coordinate descent. The L1 penalty drives some weights exactly to 0."""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X, y, x_mean, y_mean = _center(X, y)
        Xc, yc = X - x_mean, y - y_mean
        n, d = Xc.shape
        col_sq = (Xc**2).sum(axis=0)

        w = np.zeros(d)
        residual = yc.copy()  # yc - Xc @ w, kept up to date incrementally
        for _ in range(self.max_iter):
            max_change = 0.0
            for j in range(d):
                if col_sq[j] == 0:
                    continue
                old = w[j]
                rho = Xc[:, j] @ residual + col_sq[j] * old  # correlation with residual excluding feature j
                # soft-thresholding: shrink toward 0 by n * alpha, clip at 0
                w[j] = np.sign(rho) * max(abs(rho) - n * self.alpha, 0.0) / col_sq[j]
                residual -= Xc[:, j] * (w[j] - old)
                max_change = max(max_change, abs(w[j] - old))
            if max_change < self.tol:
                break

        self.weights = w
        self.bias = y_mean - x_mean @ w
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias
