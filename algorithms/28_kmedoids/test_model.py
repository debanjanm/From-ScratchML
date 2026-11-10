import numpy as np
from model import KMedoids


def _two_blobs_with_outliers():
    rng = np.random.default_rng(0)
    blob0 = rng.normal([0, 0], 0.5, (60, 2))
    blob1 = rng.normal([20, 0], 0.5, (60, 2))
    outliers = np.array([[-30.0, 0.0], [-31.0, 1.0], [-29.0, -1.0]])  # far to the left of blob0
    return np.vstack([blob0, blob1, outliers])


def test_medoids_are_actual_data_points():
    X = _two_blobs_with_outliers()
    model = KMedoids(n_clusters=2).fit(X)
    for center in model.cluster_centers_:
        assert (X == center).all(axis=1).any()
    assert np.array_equal(model.cluster_centers_, X[model.medoid_indices_])


def test_outliers_drag_the_mean_but_not_the_medoid():
    X = _two_blobs_with_outliers()
    model = KMedoids(n_clusters=2).fit(X)
    c = model.cluster_centers_[np.argmin(np.abs(model.cluster_centers_[:, 0]))]  # blob0's centre
    members = X[model.labels_ == model.predict(c[None])[0]]
    assert len(members) == 63                      # blob0 plus the 3 outliers
    assert abs(c[0]) < 1.0                          # the medoid stays inside blob0
    assert abs(members.mean(axis=0)[0]) > 1.0       # a K-Means style centroid would be dragged left


def test_manhattan_metric_and_predict():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-5, 0.5, (40, 2)), rng.normal(5, 0.5, (40, 2))])
    model = KMedoids(n_clusters=2, metric="manhattan").fit(X)
    assert len(set(model.labels_[:40])) == 1 and len(set(model.labels_[40:])) == 1
    assert model.labels_[0] != model.labels_[-1]
    assert model.predict(np.array([[-5.0, -5.0], [5.0, 5.0]])).tolist() == [model.labels_[0], model.labels_[-1]]


if __name__ == "__main__":
    test_medoids_are_actual_data_points()
    test_outliers_drag_the_mean_but_not_the_medoid()
    test_manhattan_metric_and_predict()
    print("all tests passed")
