import numpy as np
from model import LinearRegression


def test_recovers_known_line():
    rng = np.random.default_rng(0)
    X = rng.uniform(-10, 10, size=(200, 1))
    true_w, true_b = 3.5, -2.0
    y = true_w * X[:, 0] + true_b

    for method in ("normal_equation", "gradient_descent"):
        model = LinearRegression(method=method, lr=0.01, n_iters=2000).fit(X, y)
        assert abs(model.weights[0] - true_w) < 0.05, method
        assert abs(model.bias - true_b) < 0.05, method

        preds = model.predict(X)
        assert np.allclose(preds, y, atol=0.1), method


def test_multivariate():
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, size=(300, 3))
    true_w = np.array([1.0, -2.0, 0.5])
    true_b = 4.0
    y = X @ true_w + true_b

    model = LinearRegression(method="normal_equation").fit(X, y)
    assert np.allclose(model.weights, true_w, atol=1e-6)
    assert abs(model.bias - true_b) < 1e-6


if __name__ == "__main__":
    test_recovers_known_line()
    test_multivariate()
    print("all tests passed")
