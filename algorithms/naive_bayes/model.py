import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes: per-class feature mean/variance, assumes feature independence."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        epsilon = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i] = X_c.mean(axis=0)
            self.var[i] = X_c.var(axis=0) + epsilon
            self.priors[i] = X_c.shape[0] / X.shape[0]
        return self

    def _log_likelihood(self, X):
        # log N(x; mean, var) summed over features, for each class -> (n_samples, n_classes)
        log_probs = []
        for i in range(len(self.classes)):
            log_prior = np.log(self.priors[i])
            diff = X - self.mean[i]
            log_gauss = -0.5 * np.sum(np.log(2 * np.pi * self.var[i])) - 0.5 * np.sum(
                (diff**2) / self.var[i], axis=1
            )
            log_probs.append(log_prior + log_gauss)
        return np.array(log_probs).T

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        log_probs = self._log_likelihood(X)
        return self.classes[np.argmax(log_probs, axis=1)]
