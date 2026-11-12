import numpy as np
from model import LocalOutlierFactor


def _two_densities(seed=0):
    """A very tight cluster, a very sparse one, and one point that is only moderately far from the tight one."""
    rng = np.random.default_rng(seed)
    tight = rng.normal(0, 0.1, size=(100, 2))
    sparse = rng.normal([20, 0], 2.0, size=(100, 2))
    local_outlier = np.array([[1.0, 0.0]])
    return np.vstack([tight, sparse, local_outlier])


def test_inliers_score_about_one():
    X = np.random.default_rng(0).normal(size=(300, 2))
    lof = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_
    assert 0.9 < np.median(lof) < 1.1


def test_catches_local_outlier_that_a_global_distance_rule_misses():
    X = _two_densities()
    i = len(X) - 1
    model = LocalOutlierFactor(n_neighbors=10).fit(X)
    lof = -model.negative_outlier_factor_
    assert lof.argmax() == i and lof[i] > 3
    # naive rule "distance to k-th neighbour": the sparse cluster's own members look MORE isolated
    assert (model.k_distance_ > model.k_distance_[i]).sum() > 20


def test_new_points_scored_against_training_set():
    X = _two_densities()
    model = LocalOutlierFactor(n_neighbors=10, contamination=0.02).fit(X)
    new = np.array([[0.0, 0.0], [20.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    assert model.predict(new).tolist() == [1, 1, -1, -1]


def test_fit_predict_flags_contamination_fraction():
    X = np.random.default_rng(1).normal(size=(200, 2))
    labels = LocalOutlierFactor(n_neighbors=15, contamination=0.05).fit_predict(X)
    assert abs((labels == -1).mean() - 0.05) < 0.01


if __name__ == "__main__":
    test_inliers_score_about_one()
    test_catches_local_outlier_that_a_global_distance_rule_misses()
    test_new_points_scored_against_training_set()
    test_fit_predict_flags_contamination_fraction()
    print("all tests passed")
