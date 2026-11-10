import numpy as np
from model import Perceptron


def test_converges_on_separable_data_with_string_labels():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.6, (60, 2)), rng.normal(3, 0.6, (60, 2))])
    y = np.array(["a"] * 60 + ["b"] * 60)
    model = Perceptron().fit(X, y)
    assert model.errors_[-1] == 0  # converged
    assert (model.predict(X) == y).all()


def test_cannot_learn_xor():
    # the classic limit: no single line separates XOR, so it never converges
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = Perceptron(epochs=200).fit(X, y)
    assert model.errors_[-1] > 0
    assert (model.predict(X) == y).mean() < 1.0


def test_rejects_multiclass():
    try:
        Perceptron().fit(np.zeros((3, 2)), np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_converges_on_separable_data_with_string_labels()
    test_cannot_learn_xor()
    test_rejects_multiclass()
    print("all tests passed")
