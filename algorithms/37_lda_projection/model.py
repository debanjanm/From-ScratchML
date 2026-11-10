import numpy as np


class LDAProjection:
    """Fisher's Linear Discriminant Analysis as a SUPERVISED dimensionality reduction.

    PCA keeps the directions of largest variance and ignores the labels. LDA uses the labels: it
    finds directions w that maximise  (between-class scatter) / (within-class scatter) =
    w'Sb w / w'Sw w, i.e. directions along which the class means are far apart relative to how
    spread out each class is.

    With C classes there are at most C-1 useful directions. They solve the generalized eigenproblem
    Sb w = lambda Sw w; here it is turned into an ordinary symmetric one with the Cholesky factor
    Sw = L L': eigenvectors v of L^-1 Sb L^-T give w = L^-T v, scaled so w' Sw w = 1.

    (14_discriminant_analysis uses the same maths as a classifier; this one is the projection.)"""

    def __init__(self, n_components=None, reg=0.0):
        self.n_components = n_components  # None -> C - 1
        self.reg = reg                     # ridge (relative to trace) if Sw is singular, e.g. 1e-6

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        d = X.shape[1]
        k = self.n_components or len(classes) - 1
        if k > min(len(classes) - 1, d):
            raise ValueError("n_components cannot exceed min(n_classes - 1, n_features)")

        self.mean_ = X.mean(axis=0)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        for c in classes:
            Xc = X[y == c]
            mean_c = Xc.mean(axis=0)
            Sw += (Xc - mean_c).T @ (Xc - mean_c)
            diff = (mean_c - self.mean_)[:, None]
            Sb += len(Xc) * (diff @ diff.T)
        Sw += self.reg * np.trace(Sw) / d * np.eye(d)

        L = np.linalg.cholesky(Sw)
        L_inv = np.linalg.inv(L)
        eigenvalues, V = np.linalg.eigh(L_inv @ Sb @ L_inv.T)
        order = np.argsort(eigenvalues)[::-1][:k]
        self.eigenvalues_ = eigenvalues[order]  # Fisher ratio achieved by each direction
        self.explained_variance_ratio_ = self.eigenvalues_ / eigenvalues.clip(min=0).sum()
        self.components_ = (L_inv.T @ V[:, order]).T  # rows are the discriminant directions
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
