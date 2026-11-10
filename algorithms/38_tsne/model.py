import numpy as np


def _sq_dists(X):
    s = (X ** 2).sum(axis=1)
    return np.maximum(s[:, None] + s[None, :] - 2 * X @ X.T, 0.0)


def _conditional_probabilities(d, perplexity):
    """Neighbour distribution of one point given its squared distances `d` to all the others.

    A Gaussian with its own bandwidth beta = 1 / 2 sigma^2, found by binary search so that the
    entropy of the distribution is log(perplexity) -- "perplexity" = effective number of neighbours."""
    target = np.log(perplexity)
    beta, lo, hi = 1.0, 0.0, np.inf
    for _ in range(100):
        p = np.exp(-(d - d.min()) * beta)  # shift by min for numerical stability
        p /= p.sum()
        entropy = -(p * np.log(np.maximum(p, 1e-300))).sum()
        if abs(entropy - target) < 1e-5:
            break
        if entropy > target:   # too flat -> narrow the Gaussian
            lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
        else:
            hi, beta = beta, beta / 2 if lo == 0 else (beta + lo) / 2
    return p


def _joint_probabilities(X, perplexity):
    """P[i, j]: how likely i and j are neighbours in the original space (symmetrized, sums to 1)."""
    n = len(X)
    D = _sq_dists(X)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, np.arange(n) != i] = _conditional_probabilities(np.delete(D[i], i), perplexity)
    P = P + P.T
    return P / P.sum()


def _kl_and_grad(Y, P):
    """KL(P || Q) where Q uses a heavy-tailed Student-t kernel in the low-dim space, and its gradient."""
    num = 1.0 / (1.0 + _sq_dists(Y))
    np.fill_diagonal(num, 0.0)
    Q = num / num.sum()
    kl = (P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12))).sum()
    W = (P - Q) * num
    grad = 4.0 * (W.sum(axis=1)[:, None] * Y - W @ Y)
    return kl, grad


class TSNE:
    """t-distributed Stochastic Neighbor Embedding (exact version, van der Maaten & Hinton 2008).

    Finds 2-D (or k-D) points whose neighbour structure matches the high-dimensional data:
    1. P = neighbour probabilities in the original space (Gaussians sized by `perplexity`).
    2. Q = neighbour probabilities of the embedding, with a Student-t kernel 1/(1+d^2). Its heavy
       tail lets dissimilar points sit far apart, which fixes the "crowding problem".
    3. Move the points by gradient descent (momentum + per-parameter gains) to minimise KL(P || Q).
       The first `exaggeration_iters` steps multiply P by 12 so clusters form tight and well apart.

    Use it to LOOK at data. Distances between far-apart clusters and cluster sizes are not
    meaningful, there is no transform() for new points, and different seeds give different pictures.

    ponytail: exact O(n^2) per iteration (n up to a few thousand); Barnes-Hut / FFT variants for more."""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate="auto", n_iter=1000,
                 exaggeration=12.0, exaggeration_iters=250, random_state=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.exaggeration = exaggeration
        self.exaggeration_iters = exaggeration_iters
        self.random_state = random_state

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        P = _joint_probabilities(X, min(self.perplexity, (n - 1) / 3))
        lr = max(n / self.exaggeration / 4, 50) if self.learning_rate == "auto" else self.learning_rate

        rng = np.random.default_rng(self.random_state)
        Y = 1e-4 * rng.standard_normal((n, self.n_components))
        update = np.zeros_like(Y)
        gains = np.ones_like(Y)
        for it in range(self.n_iter):
            early = it < self.exaggeration_iters
            _, grad = _kl_and_grad(Y, P * self.exaggeration if early else P)
            gains = np.maximum(np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8), 0.01)
            update = (0.5 if early else 0.8) * update - lr * gains * grad
            Y = Y + update
            Y -= Y.mean(axis=0)

        self.embedding_ = Y
        self.kl_divergence_ = _kl_and_grad(Y, P)[0]
        return Y

    def fit(self, X):
        self.fit_transform(X)
        return self
