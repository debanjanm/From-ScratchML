import numpy as np


class PCA:
    """Principal Component Analysis via eigendecomposition of the covariance matrix."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components = eigenvectors[:, : self.n_components].T
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / total_variance
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean) @ self.components.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
