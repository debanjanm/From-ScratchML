import numpy as np
from model import DBSCAN


def test_two_blobs_and_an_outlier():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (40, 2)), rng.normal(5, 0.3, (40, 2)), [[20.0, 20.0]]])
    labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)

    assert labels[-1] == -1  # the far point is noise
    assert len(set(labels[:40])) == 1 and len(set(labels[40:80])) == 1
    assert labels[0] != labels[40]
    assert set(labels) == {-1, 0, 1}


def test_separates_two_moons_which_kmeans_cannot():
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
    labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
    clusters = [c for c in set(labels) if c != -1]
    assert len(clusters) == 2
    # each moon should be (almost) a single cluster
    for moon in (0, 1):
        counts = np.bincount(labels[y == moon][labels[y == moon] != -1])
        assert counts.max() / (y == moon).sum() > 0.95


def test_all_noise_when_min_samples_too_high():
    X = np.arange(10.0).reshape(-1, 1)
    assert (DBSCAN(eps=0.5, min_samples=3).fit_predict(X) == -1).all()


if __name__ == "__main__":
    test_two_blobs_and_an_outlier()
    test_separates_two_moons_which_kmeans_cannot()
    test_all_noise_when_min_samples_too_high()
    print("all tests passed")
