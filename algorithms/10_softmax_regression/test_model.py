import numpy as np
from model import SoftmaxRegression, _softmax


def test_softmax_rows_sum_to_one_and_survive_huge_logits():
    P = _softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.isfinite(P).all()
    assert np.allclose(P[1], 1 / 3)


def test_three_blobs_with_string_labels():
    rng = np.random.default_rng(0)
    centers = np.array([[-4, 0], [4, 0], [0, 5]])
    X = np.vstack([rng.normal(c, 0.7, (50, 2)) for c in centers])
    y = np.repeat(np.array(["red", "green", "blue"]), 50)

    model = SoftmaxRegression(lr=0.2, n_iters=500).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.98
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_two_classes_reduces_to_logistic_regression():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-2, 1, (80, 2)), rng.normal(2, 1, (80, 2))])
    y = np.array([0] * 80 + [1] * 80)
    assert (SoftmaxRegression(n_iters=500).fit(X, y).predict(X) == y).mean() > 0.95


if __name__ == "__main__":
    test_softmax_rows_sum_to_one_and_survive_huge_logits()
    test_three_blobs_with_string_labels()
    test_two_classes_reduces_to_logistic_regression()
    print("all tests passed")
