import numpy as np


def _log_gaussian(X, mean, cov):
    """log N(x | mean, cov) for every row of X."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)  # whitened differences
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + log_det + (z**2).sum(axis=0))


def _logsumexp(a):
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))).ravel()


class GaussianMixture:
    """Soft clustering: the data is modelled as k Gaussians, each with its own weight, mean and full
    covariance. Fitted by Expectation-Maximization:

    E-step: for every point compute the RESPONSIBILITY of each component (its posterior probability).
    M-step: re-estimate every component's weight, mean and covariance from the responsibility-weighted data.

    Each iteration is guaranteed not to decrease the log-likelihood (log_likelihood_history_ shows it).
    Unlike K-Means, points belong to clusters with a probability, and clusters can be elliptical.

    reg_covar is added to every covariance diagonal so a component can't collapse onto a point.
    ponytail: single k-means style initialization, not several restarts; EM only finds a local optimum."""

    def __init__(self, n_components=3, max_iter=200, tol=1e-8, reg_covar=1e-6, random_state=0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self._m_step(X, self._initial_responsibilities(X, rng))

        self.log_likelihood_history_ = []
        previous = -np.inf
        for _ in range(self.max_iter):
            log_norm, resp = self._e_step(X)
            ll = log_norm.mean()
            self.log_likelihood_history_.append(ll)
            self._m_step(X, resp)
            if abs(ll - previous) < self.tol:
                break
            previous = ll
        self.labels_ = self.predict(X)
        return self

    def _initial_responsibilities(self, X, rng):
        # k-means++ seeds, a few Lloyd steps, then hard assignments as the starting responsibilities
        n, k = len(X), self.n_components
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
            centers.append(X[rng.choice(n, p=d2 / d2.sum())])
        centers = np.array(centers)
        for _ in range(10):
            labels = ((X[:, None] - centers[None]) ** 2).sum(-1).argmin(axis=1)
            centers = np.array([X[labels == c].mean(axis=0) if (labels == c).any() else centers[c] for c in range(k)])
        return np.eye(k)[((X[:, None] - centers[None]) ** 2).sum(-1).argmin(axis=1)]

    def _e_step(self, X):
        log_prob = np.column_stack([
            np.log(w) + _log_gaussian(X, m, c)
            for w, m, c in zip(self.weights_, self.means_, self.covariances_)
        ])
        log_norm = _logsumexp(log_prob)  # log p(x) for each point
        return log_norm, np.exp(log_prob - log_norm[:, None])

    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(axis=0) + 1e-12  # effective number of points in each component
        self.weights_ = nk / n
        self.means_ = (resp.T @ X) / nk[:, None]
        self.covariances_ = np.array([
            ((resp[:, k, None] * (X - self.means_[k])).T @ (X - self.means_[k])) / nk[k]
            + self.reg_covar * np.eye(d)
            for k in range(self.n_components)
        ])

    def predict_proba(self, X):
        return self._e_step(np.asarray(X, dtype=float))[1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X):
        """Average log-likelihood per sample."""
        return self._e_step(np.asarray(X, dtype=float))[0].mean()

    def bic(self, X):
        """Bayesian Information Criterion: lower is better; penalizes extra components."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_components
        n_params = k * d + k * d * (d + 1) // 2 + (k - 1)
        return -2.0 * self.score(X) * n + n_params * np.log(n)
