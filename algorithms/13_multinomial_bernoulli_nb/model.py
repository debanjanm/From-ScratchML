import numpy as np


class _NaiveBayesBase:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing: pretend every feature was seen alpha extra times
        self.classes_ = None
        self.log_prior = None
        self.log_prob = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.log_prior = np.log(np.bincount(y_enc) / len(y_enc))
        self._fit_likelihood(X, y_enc)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


class MultinomialNB(_NaiveBayesBase):
    """For count features (word counts, histograms). P(feature j | class c) is the smoothed
    share of class c's total count that falls on feature j."""

    def _fit_likelihood(self, X, y_enc):
        counts = np.array([X[y_enc == c].sum(axis=0) for c in range(len(self.classes_))]) + self.alpha
        self.log_prob = np.log(counts / counts.sum(axis=1, keepdims=True))

    def _joint_log_likelihood(self, X):
        return X @ self.log_prob.T + self.log_prior  # a feature seen x times contributes x * log p


class BernoulliNB(_NaiveBayesBase):
    """For binary features (word present / absent). Unlike Multinomial, the *absence* of a
    feature is evidence too. Features must be 0/1."""

    def _fit_likelihood(self, X, y_enc):
        n_c = np.bincount(y_enc)
        ones = np.array([X[y_enc == c].sum(axis=0) for c in range(len(self.classes_))])
        self.log_prob = np.log((ones + self.alpha) / (n_c[:, None] + 2 * self.alpha))  # P(x_j = 1 | c)
        self.log_neg_prob = np.log(1 - np.exp(self.log_prob))  # P(x_j = 0 | c)

    def _joint_log_likelihood(self, X):
        return X @ self.log_prob.T + (1 - X) @ self.log_neg_prob.T + self.log_prior
