import numpy as np


class ElasticNet:
    """Least squares + a mix of L1 and L2 penalties (scikit-learn's convention):

        1/(2n) * ||y - Xw||^2  +  alpha * l1_ratio * ||w||_1  +  0.5 * alpha * (1 - l1_ratio) * ||w||^2

    l1_ratio=1 is Lasso, l1_ratio=0 is Ridge. Solved by cyclic coordinate descent.
    The bias is not penalized (data is centered, bias recovered from the means)."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-8):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        n, d = Xc.shape
        col_sq = (Xc**2).sum(axis=0)
        l1 = n * self.alpha * self.l1_ratio
        l2 = n * self.alpha * (1 - self.l1_ratio)

        w = np.zeros(d)
        residual = yc.copy()  # yc - Xc @ w, kept up to date incrementally
        for _ in range(self.max_iter):
            max_change = 0.0
            for j in range(d):
                if col_sq[j] == 0:
                    continue
                old = w[j]
                rho = Xc[:, j] @ residual + col_sq[j] * old  # correlation with residual excluding j
                # soft-threshold by the L1 part; the L2 part just inflates the denominator
                w[j] = np.sign(rho) * max(abs(rho) - l1, 0.0) / (col_sq[j] + l2)
                residual -= Xc[:, j] * (w[j] - old)
                max_change = max(max_change, abs(w[j] - old))
            if max_change < self.tol:
                break

        self.weights = w
        self.bias = y_mean - x_mean @ w
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias
