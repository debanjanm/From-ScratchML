import numpy as np
from model import MeanShift, estimate_bandwidth


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])
    X = np.vstack([rng.normal(c, 0.6, (50, 2)) for c in centers])
    return X, centers


def test_finds_the_number_of_clusters_by_itself():
    X, centers = _blobs()
    model = MeanShift(bandwidth=2.0).fit(X)
    assert len(model.cluster_centers_) == 3          # nobody told it k = 3
    for c in centers:                                 # every true centre has a found mode nearby
        assert np.linalg.norm(model.cluster_centers_ - c, axis=1).min() < 0.5
    assert [len(set(model.labels_[i * 50:(i + 1) * 50])) for i in range(3)] == [1, 1, 1]


def test_bandwidth_controls_granularity():
    X, _ = _blobs()
    small = len(MeanShift(bandwidth=0.4).fit(X).cluster_centers_)
    right = len(MeanShift(bandwidth=2.0).fit(X).cluster_centers_)
    huge = len(MeanShift(bandwidth=30.0).fit(X).cluster_centers_)
    assert small > right == 3 > huge == 1


def test_default_bandwidth_is_estimated_and_predict_assigns_new_points():
    X, centers = _blobs(1)
    model = MeanShift().fit(X)
    assert model.bandwidth_ == estimate_bandwidth(X)
    near_each_center = model.predict(centers + 0.1)
    assert len(set(near_each_center)) == len(near_each_center) == 3


if __name__ == "__main__":
    test_finds_the_number_of_clusters_by_itself()
    test_bandwidth_controls_granularity()
    test_default_bandwidth_is_estimated_and_predict_assigns_new_points()
    print("all tests passed")
