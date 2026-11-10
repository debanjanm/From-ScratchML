from itertools import combinations_with_replacement

import numpy as np


def polynomial_features(X, degree):
    """All monomials of the features up to `degree` (no constant column).
    For features a, b and degree 2: a, b, a^2, a*b, b^2."""
    X = np.asarray(X, dtype=float)
    cols = [
        np.prod(X[:, combo], axis=1)
        for d in range(1, degree + 1)
        for combo in combinations_with_replacement(range(X.shape[1]), d)
    ]
    return np.column_stack(cols)


class PolynomialRegression:
    """Linear regression on polynomial features: still linear in the weights, curved in x.

    ponytail: no feature scaling, high degrees on wide-range data get ill-conditioned.
    Standardize X first or add a ridge penalty if that bites."""

    def __init__(self, degree=2):
        self.degree = degree
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        P = polynomial_features(X, self.degree)
        A = np.hstack([np.ones((len(P), 1)), P])
        theta, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        self.bias, self.weights = theta[0], theta[1:]
        return self

    def predict(self, X):
        return polynomial_features(X, self.degree) @ self.weights + self.bias
