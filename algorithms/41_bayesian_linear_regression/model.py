import numpy as np


class BayesianLinearRegression:
    """Linear regression that returns a DISTRIBUTION over weights, not one point estimate.

    Model:  y = X w + noise,   noise ~ N(0, 1/alpha),   w ~ N(0, 1/lambda * I)
    (alpha = noise precision, lambda = weight precision; same names as scikit-learn's BayesianRidge.)

    With those Gaussians the posterior over w is Gaussian in closed form:
        S = (lambda I + alpha X'X)^-1          (posterior covariance)
        m = alpha S X'y                        (posterior mean -- equals ridge regression)
    and the prediction at x is Gaussian too:
        mean = x.m          variance = 1/alpha + x' S x
    i.e. noise in the data + uncertainty about the weights (which grows away from the training data).

    alpha and lambda are NOT hand-tuned: they are learned by maximising the marginal likelihood
    (evidence) with MacKay's fixed-point updates (Bishop, PRML 3.5). Overfitting is controlled
    automatically -- no cross-validation needed.

    ponytail: fixed-point iteration with a flat (improper-ish) prior on alpha, lambda; uses the SVD of X
    once so every iteration is O(d)."""

    def __init__(self, max_iter=300, tol=1e-3, alpha_init=None, lambda_init=1.0, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.x_mean_ = X.mean(axis=0) if self.fit_intercept else np.zeros(d)
        self.y_mean_ = y.mean() if self.fit_intercept else 0.0
        Xc, yc = X - self.x_mean_, y - self.y_mean_

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        eig = s ** 2                            # eigenvalues of X'X
        Uty = U.T @ yc
        resid_outside = yc @ yc - Uty @ Uty     # part of y no linear combination of X can reach

        alpha = self.alpha_init or 1.0 / (np.var(yc) + 1e-12)
        lam = self.lambda_init
        for _ in range(self.max_iter):
            m = Vt.T @ (alpha * s / (lam + alpha * eig) * Uty)
            gamma = (alpha * eig / (lam + alpha * eig)).sum()  # effective number of well-determined weights
            rss = ((yc - Xc @ m) ** 2).sum()
            lam_new = gamma / max(m @ m, 1e-12)
            alpha_new = (n - gamma) / max(rss, 1e-12)
            done = abs(lam_new - lam) < self.tol * lam and abs(alpha_new - alpha) < self.tol * alpha
            lam, alpha = lam_new, alpha_new
            if done:
                break

        self.alpha_, self.lambda_ = alpha, lam
        self.sigma_ = Vt.T @ np.diag(1.0 / (lam + alpha * eig)) @ Vt   # posterior covariance of w
        self.coef_ = alpha * self.sigma_ @ Xc.T @ yc                   # posterior mean of w
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        mean = X @ self.coef_ + self.intercept_
        if not return_std:
            return mean
        Xc = X - self.x_mean_
        var = 1.0 / self.alpha_ + ((Xc @ self.sigma_) * Xc).sum(axis=1)
        return mean, np.sqrt(var)
