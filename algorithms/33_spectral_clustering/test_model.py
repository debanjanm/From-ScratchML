import numpy as np
from model import SpectralClustering


def _moons(seed=0, n=200):
    from sklearn.datasets import make_moons

    return make_moons(n_samples=n, noise=0.05, random_state=seed)


def test_separates_interleaved_moons_where_kmeans_fails():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    X, truth = _moons()
    spectral = SpectralClustering(n_clusters=2).fit_predict(X)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    assert adjusted_rand_score(truth, spectral) > 0.95
    assert adjusted_rand_score(truth, kmeans) < 0.6      # round-cluster assumption breaks on moons


def test_separates_concentric_circles_with_rbf_affinity():
    from sklearn.datasets import make_circles
    from sklearn.metrics import adjusted_rand_score

    X, truth = make_circles(n_samples=200, factor=0.4, noise=0.04, random_state=0)
    labels = SpectralClustering(n_clusters=2, affinity="rbf", gamma=20.0).fit_predict(X)
    assert adjusted_rand_score(truth, labels) > 0.95


def test_affinity_is_symmetric_and_embedding_rows_are_unit_length():
    X, _ = _moons(n=80)
    model = SpectralClustering(n_clusters=3).fit(X)
    W = model.affinity_matrix_
    assert np.allclose(W, W.T) and (np.diag(W) == 0).all()
    assert np.allclose(np.linalg.norm(model.embedding_, axis=1), 1.0)
    assert len(set(model.labels_)) == 3


if __name__ == "__main__":
    test_separates_interleaved_moons_where_kmeans_fails()
    test_separates_concentric_circles_with_rbf_affinity()
    test_affinity_is_symmetric_and_embedding_rows_are_unit_length()
    print("all tests passed")
