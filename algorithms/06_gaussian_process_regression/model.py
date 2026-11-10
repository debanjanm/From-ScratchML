import numpy as np


def _rbf_kernel(A, B, length_scale, variance):
    sq_dist = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return variance * np.exp(-0.5 * sq_dist / length_scale**2)


class GaussianProcessRegressor:
    """Gaussian process with an RBF kernel and a zero-mean prior. Predicts a mean *and* an
    uncertainty (std) for every point: wide away from the data, tight near it.

    noise is the std of the observation noise. Hyperparameters are fixed, not learned.

    ponytail: O(n^3) fit, O(n^2) memory, no marginal-likelihood tuning of the hyperparameters."""

    def __init__(self, length_scale=1.0, variance=1.0, noise=0.1):
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.X_train = None
        self.L = None
        self.alpha = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        K = _rbf_kernel(self.X_train, self.X_train, self.length_scale, self.variance)
        K += self.noise**2 * np.eye(len(y))
        self.L = np.linalg.cholesky(K)  # K = L L^T, stable way to "invert" K
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))  # K^-1 y
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        K_star = _rbf_kernel(X, self.X_train, self.length_scale, self.variance)
        mean = K_star @ self.alpha
        if not return_std:
            return mean
        v = np.linalg.solve(self.L, K_star.T)
        var = self.variance - (v**2).sum(axis=0)  # prior variance minus what the data explains
        return mean, np.sqrt(np.clip(var, 0.0, None))
