import numpy as np
from model import LogisticRegression, _sigmoid


def test_sigmoid_bounds():
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(100) > 0.999
    assert _sigmoid(-100) < 0.001


def test_separable_blobs():
    rng = np.random.default_rng(0)
    class0 = rng.normal(loc=-3, scale=0.5, size=(100, 2))
    class1 = rng.normal(loc=3, scale=0.5, size=(100, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 100 + [1] * 100)

    model = LogisticRegression(lr=0.1, n_iters=2000).fit(X, y)
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.98, accuracy


if __name__ == "__main__":
    test_sigmoid_bounds()
    test_separable_blobs()
    print("all tests passed")
