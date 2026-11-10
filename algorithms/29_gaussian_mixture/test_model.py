import numpy as np
from model import GaussianMixture


def _mixture(n=900, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [int(n * 0.2), int(n * 0.3), n - int(n * 0.2) - int(n * 0.3)]
    covs = [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.2], [1.2, 1.0]], [[0.5, 0.0], [0.0, 2.0]]]
    means = [[-6, 0], [0, 6], [6, -2]]
    X = np.vstack([rng.multivariate_normal(m, c, s) for m, c, s in zip(means, covs, sizes)])
    return X, np.repeat([0, 1, 2], sizes)


def test_log_likelihood_never_decreases():
    X, _ = _mixture()
    ll = GaussianMixture(n_components=3, tol=0).fit(X).log_likelihood_history_
    assert len(ll) > 3
    assert all(b >= a - 1e-9 for a, b in zip(ll, ll[1:]))  # the defining property of EM


def test_recovers_the_true_parameters():
    X, truth = _mixture()
    model = GaussianMixture(n_components=3).fit(X)
    order = np.argsort(model.means_[:, 0])  # components sorted left to right
    assert np.allclose(model.weights_[order], [0.2, 0.3, 0.5], atol=0.04)
    assert np.allclose(model.means_[order], [[-6, 0], [0, 6], [6, -2]], atol=0.3)
    assert abs(model.covariances_[order][1][0, 1] - 1.2) < 0.3   # learned a tilted (elliptical) shape
    pred = model.predict(X)  # label ids are arbitrary, so score by purity instead of equality
    assert sum(np.bincount(pred[truth == c]).max() for c in range(3)) / len(X) > 0.95


def test_probabilities_and_bic_picks_the_right_number_of_components():
    X, _ = _mixture()
    probs = GaussianMixture(n_components=3).fit(X).predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0)
    bics = {k: GaussianMixture(n_components=k).fit(X).bic(X) for k in range(1, 6)}
    assert min(bics, key=bics.get) == 3


if __name__ == "__main__":
    test_log_likelihood_never_decreases()
    test_recovers_the_true_parameters()
    test_probabilities_and_bic_picks_the_right_number_of_components()
    print("all tests passed")
