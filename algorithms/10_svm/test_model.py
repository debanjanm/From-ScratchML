import numpy as np
from model import LinearSVM


def test_separable_blobs_string_labels():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.6, (80, 2)), rng.normal(3, 0.6, (80, 2))])
    y = np.array(["neg"] * 80 + ["pos"] * 80)

    model = LinearSVM().fit(X, y)
    assert (model.predict(X) == y).all()
    assert model.weights @ np.array([1.0, 1.0]) > 0  # boundary orientation: "pos" is up-right


def test_margin_is_respected():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-4, 0.5, (60, 2)), rng.normal(4, 0.5, (60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    model = LinearSVM(n_iters=2000).fit(X, y)
    scores = model.decision_function(X) * (2 * y - 1)
    assert scores.min() > 0.9  # separable data ends up (almost) outside the margin


def test_rejects_multiclass():
    try:
        LinearSVM().fit(np.zeros((3, 2)), np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_separable_blobs_string_labels()
    test_margin_is_respected()
    test_rejects_multiclass()
    print("all tests passed")
