import numpy as np


class _GaussianDiscriminant:
    def _setup(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.priors = np.bincount(y_enc) / len(y_enc)
        self.means = np.array([X[y_enc == c].mean(axis=0) for c in range(len(self.classes_))])
        return X, y_enc

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(np.asarray(X, dtype=float)), axis=1)]


class LinearDiscriminantAnalysis(_GaussianDiscriminant):
    """Each class is a Gaussian; all classes share ONE covariance matrix. The log-posterior is then
    linear in x, so decision boundaries are straight lines (hyperplanes):

        score_k(x) = x . S^-1 mu_k  -  0.5 * mu_k . S^-1 mu_k  +  log prior_k"""

    def fit(self, X, y):
        X, y_enc = self._setup(X, y)
        centered = X - self.means[y_enc]
        pooled_cov = centered.T @ centered / (len(X) - len(self.classes_))  # within-class covariance
        self.coef = np.linalg.solve(pooled_cov, self.means.T).T  # S^-1 mu_k, one row per class
        self.intercept = -0.5 * np.sum(self.coef * self.means, axis=1) + np.log(self.priors)
        return self

    def _scores(self, X):
        return X @ self.coef.T + self.intercept


class QuadraticDiscriminantAnalysis(_GaussianDiscriminant):
    """Each class gets its OWN covariance matrix, so boundaries are curved (quadratic):

        score_k(x) = -0.5 * log|S_k| - 0.5 * (x - mu_k)' S_k^-1 (x - mu_k) + log prior_k

    reg adds reg * I to every covariance (needed when a class has fewer samples than features)."""

    def __init__(self, reg=0.0):
        self.reg = reg

    def fit(self, X, y):
        X, y_enc = self._setup(X, y)
        d = X.shape[1]
        self.covs = [np.cov(X[y_enc == c], rowvar=False) + self.reg * np.eye(d) for c in range(len(self.classes_))]
        return self

    def _scores(self, X):
        scores = []
        for mu, cov, prior in zip(self.means, self.covs, self.priors):
            diff = X - mu
            maha = np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)  # squared Mahalanobis distance
            scores.append(-0.5 * np.linalg.slogdet(cov)[1] - 0.5 * maha + np.log(prior))
        return np.array(scores).T
