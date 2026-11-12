import numpy as np
from model import IsolationForest, _c


def _data(seed=0):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(size=(500, 2))
    outliers = np.sign(rng.normal(size=(15, 2))) * rng.uniform(5, 9, size=(15, 2))
    return np.vstack([inliers, outliers]), np.array([0] * 500 + [1] * 15)


def test_average_path_length_known_values():
    assert _c(1) == 0
    assert _c(2) == 1
    assert np.isclose(_c(256), 2 * (np.log(255) + 0.5772156649) - 2 * 255 / 256)
    assert _c(256) > _c(16) > _c(2)                              # grows like log n


def test_outliers_score_higher_and_are_ranked_first():
    X, is_out = _data()
    model = IsolationForest(random_state=0).fit(X)
    s = model.score_samples(X)
    assert ((s > 0) & (s < 1)).all()
    assert s[is_out == 1].min() > s[is_out == 0].mean()
    assert s[is_out == 1].mean() > 0.6 and s[is_out == 0].mean() < 0.5
    top = np.argsort(s)[::-1][: is_out.sum()]
    assert is_out[top].mean() > 0.9   # top-15 scores are (almost) exactly the 15 planted outliers


def test_predict_flags_contamination_fraction_and_is_reproducible():
    X, _ = _data()
    labels = IsolationForest(contamination=0.05, random_state=3).fit_predict(X)
    assert set(labels) == {-1, 1}
    assert abs((labels == -1).mean() - 0.05) < 0.01
    again = IsolationForest(contamination=0.05, random_state=3).fit_predict(X)
    assert (labels == again).all()


def test_far_new_point_is_an_anomaly_center_point_is_not():
    X, _ = _data()
    model = IsolationForest(random_state=0).fit(X)
    assert model.predict([[0.0, 0.0], [40.0, 40.0]]).tolist() == [1, -1]


if __name__ == "__main__":
    test_average_path_length_known_values()
    test_outliers_score_higher_and_are_ranked_first()
    test_predict_flags_contamination_fraction_and_is_reproducible()
    test_far_new_point_is_an_anomaly_center_point_is_not()
    print("all tests passed")
