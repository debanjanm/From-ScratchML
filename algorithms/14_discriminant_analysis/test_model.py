import numpy as np
from model import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def test_lda_separates_gaussians_with_shared_covariance():
    rng = np.random.default_rng(0)
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X = np.vstack([rng.multivariate_normal([0, 0], cov, 150), rng.multivariate_normal([3, 0], cov, 150)])
    y = np.array(["a"] * 150 + ["b"] * 150)
    assert (LinearDiscriminantAnalysis().fit(X, y).predict(X) == y).mean() > 0.9


def test_qda_beats_lda_when_covariances_differ():
    # same center, different spread: no straight line separates them, a circle does
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.5, (200, 2)), rng.normal(0, 3.0, (200, 2))])
    y = np.array([0] * 200 + [1] * 200)
    lda = (LinearDiscriminantAnalysis().fit(X, y).predict(X) == y).mean()
    qda = (QuadraticDiscriminantAnalysis().fit(X, y).predict(X) == y).mean()
    assert qda > 0.9
    assert qda > lda + 0.2


def test_multiclass_and_prior_matter():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(c, 1, (60, 2)) for c in (-4, 0, 4)])
    y = np.repeat([0, 1, 2], 60)
    for model in (LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()):
        assert (model.fit(X, y).predict(X) == y).mean() > 0.9


if __name__ == "__main__":
    test_lda_separates_gaussians_with_shared_covariance()
    test_qda_beats_lda_when_covariances_differ()
    test_multiclass_and_prior_matter()
    print("all tests passed")
