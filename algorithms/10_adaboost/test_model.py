import numpy as np
from model import AdaBoostClassifier, _Stump


def test_stump_finds_the_split():
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    stump = _Stump().fit(X, y, np.full(6, 1 / 6))
    assert (stump.predict(X) == y).all()
    assert 3 < stump.threshold < 8


def test_boosting_beats_a_single_stump():
    # positive class is the middle band: no single threshold can capture it
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(300, 1))
    y = np.where((X[:, 0] > 3) & (X[:, 0] < 7), "in", "out")

    one = AdaBoostClassifier(n_estimators=1).fit(X, y)
    many = AdaBoostClassifier(n_estimators=30).fit(X, y)
    acc = lambda m: (m.predict(X) == y).mean()
    assert acc(one) < 0.85
    assert acc(many) > 0.95


def test_rejects_multiclass():
    try:
        AdaBoostClassifier().fit(np.zeros((3, 2)), np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_stump_finds_the_split()
    test_boosting_beats_a_single_stump()
    test_rejects_multiclass()
    print("all tests passed")
