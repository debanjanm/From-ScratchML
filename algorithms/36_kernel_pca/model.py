import numpy as np


class KernelPCA:
    """PCA in an implicit high-dimensional feature space, via the kernel trick.

    Plain PCA only finds straight directions. Kernel PCA maps points through phi(x) (never computed)
    and does PCA there, needing only the kernel k(a, b) = phi(a) . phi(b):

    1. Kernel matrix K (n x n); center it so the implicit features have zero mean.
    2. Eigendecompose it. A component's score for training point i is sqrt(lambda) * v[i].
    3. A new point is scored with its (identically centered) kernel row against the training set,
       scaled by 1/sqrt(lambda).

    Kernels: "linear" (== ordinary PCA), "poly" (gamma*a.b + coef0)^degree, "rbf" exp(-gamma*|a-b|^2).

    ponytail: stores the full n x n kernel matrix and eigendecomposes it, O(n^2) memory / O(n^3) time."""

    def __init__(self, n_components=2, kernel="rbf", gamma=None, degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma  # None -> 1 / n_features
        self.degree = degree
        self.coef0 = coef0

    def _kernel(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (self.gamma_ * (A @ B.T) + self.coef0) ** self.degree
        if self.kernel == "rbf":
            d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return np.exp(-self.gamma_ * np.maximum(d2, 0.0))
        raise ValueError(f"unknown kernel: {self.kernel}")

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        K = self._kernel(X, X)
        self.K_col_mean_ = K.mean(axis=0)
        self.K_mean_ = K.mean()
        Kc = K - self.K_col_mean_[None, :] - self.K_col_mean_[:, None] + self.K_mean_

        eigenvalues, eigenvectors = np.linalg.eigh(Kc)
        order = np.argsort(eigenvalues)[::-1][: self.n_components]
        self.eigenvalues_ = eigenvalues[order]
        keep = self.eigenvalues_ > 1e-10  # drop numerically-zero components
        self.eigenvalues_ = self.eigenvalues_[keep]
        self.alphas_ = eigenvectors[:, order][:, keep] / np.sqrt(self.eigenvalues_)
        self.X_fit_ = X
        return self

    def transform(self, X):
        K = self._kernel(np.asarray(X, dtype=float), self.X_fit_)
        Kc = K - K.mean(axis=1, keepdims=True) - self.K_col_mean_[None, :] + self.K_mean_
        return Kc @ self.alphas_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
