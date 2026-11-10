import numpy as np
from model import TSNE, _conditional_probabilities, _joint_probabilities, _kl_and_grad, _sq_dists


def _blobs(seed=0, per=40, dim=10, k=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8, size=(k, dim))
    X = np.vstack([c + rng.normal(size=(per, dim)) for c in centers])
    return X, np.repeat(np.arange(k), per)


def test_bandwidth_search_hits_the_requested_perplexity():
    d = np.random.default_rng(0).uniform(0.1, 50, size=99) ** 2
    for perplexity in (5, 15, 30):
        p = _conditional_probabilities(d, perplexity)
        p = p[p > 0]
        assert np.isclose(np.exp(-(p * np.log(p)).sum()), perplexity, rtol=1e-3)


def test_joint_probabilities_are_a_valid_symmetric_distribution():
    X, _ = _blobs()
    P = _joint_probabilities(X, perplexity=15)
    assert np.allclose(P, P.T) and np.isclose(P.sum(), 1.0) and (np.diag(P) == 0).all()
    D = _sq_dists(X)
    assert P[0, np.argsort(D[0])[1]] > P[0, np.argmax(D[0])]   # nearer point -> larger probability


def test_gradient_matches_finite_differences():
    X, _ = _blobs(per=8, k=2)
    P = _joint_probabilities(X, perplexity=5)
    Y = np.random.default_rng(1).normal(size=(len(X), 2))
    _, grad = _kl_and_grad(Y, P)
    numeric = np.zeros_like(Y)
    eps = 1e-6
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, j] += eps
            Ym[i, j] -= eps
            numeric[i, j] = (_kl_and_grad(Yp, P)[0] - _kl_and_grad(Ym, P)[0]) / (2 * eps)
    assert np.abs(grad - numeric).max() < 1e-7


def test_embedding_keeps_clusters_together_and_lowers_kl():
    X, y = _blobs()
    model = TSNE(perplexity=15, n_iter=500, random_state=0)
    Y = model.fit_transform(X)
    assert Y.shape == (len(X), 2)
    nearest = np.argsort(_sq_dists(Y), axis=1)[:, 1]
    assert (y[nearest] == y).mean() > 0.99             # each point's closest embedded neighbour is in its cluster
    random_kl = _kl_and_grad(np.random.default_rng(0).normal(size=Y.shape), _joint_probabilities(X, 15))[0]
    assert model.kl_divergence_ < 0.2 * random_kl      # and the objective really went down


if __name__ == "__main__":
    test_bandwidth_search_hits_the_requested_perplexity()
    test_joint_probabilities_are_a_valid_symmetric_distribution()
    test_gradient_matches_finite_differences()
    test_embedding_keeps_clusters_together_and_lowers_kl()
    print("all tests passed")
