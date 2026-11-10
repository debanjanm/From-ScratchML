import numpy as np


class TruncatedSVD:
    """Best rank-k approximation X ~ U S Vt, keeping the k largest singular values.

    Same as PCA except the data is NOT centered first, so it also works on data where zero means
    "absent" (word counts, tf-idf, ratings). Eckart-Young: no other rank-k matrix is closer to X.

    Computed from the eigendecomposition of the Gram matrix XtX: its eigenvalues are the squared
    singular values and its eigenvectors are the right singular vectors V.

    ponytail: forming XtX squares the condition number (tiny singular values lose precision) and
    costs O(n d^2); use np.linalg.svd or a randomized/Lanczos solver for big or ill-conditioned X."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(X.T @ X)
        order = np.argsort(eigenvalues)[::-1][: self.n_components]
        self.singular_values_ = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        self.components_ = eigenvectors[:, order].T  # rows are the right singular vectors
        Z = X @ self.components_.T  # = U S
        self.explained_variance_ = Z.var(axis=0)
        self.explained_variance_ratio_ = self.explained_variance_ / X.var(axis=0).sum()
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Map k-dim coordinates back to the original space (the rank-k reconstruction)."""
        return np.asarray(Z, dtype=float) @ self.components_
