import numpy as np
from model import PolynomialRegression, polynomial_features


def test_feature_expansion():
    X = np.array([[2.0, 3.0]])
    # a, b, a^2, a*b, b^2
    assert polynomial_features(X, 2).tolist() == [[2.0, 3.0, 4.0, 6.0, 9.0]]
    assert polynomial_features(X, 1).tolist() == [[2.0, 3.0]]


def test_recovers_exact_cubic():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(100, 1))
    y = 2 * X[:, 0] ** 3 - X[:, 0] ** 2 + 0.5 * X[:, 0] + 4
    model = PolynomialRegression(degree=3).fit(X, y)
    assert np.allclose(model.weights, [0.5, -1.0, 2.0], atol=1e-8)
    assert abs(model.bias - 4.0) < 1e-8


def test_higher_degree_fits_curve_better_than_a_line():
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = X[:, 0] ** 2 + rng.normal(scale=0.1, size=200)
    mse = lambda m: np.mean((m.predict(X) - y) ** 2)
    assert mse(PolynomialRegression(degree=2).fit(X, y)) < 0.05 * mse(PolynomialRegression(degree=1).fit(X, y))


if __name__ == "__main__":
    test_feature_expansion()
    test_recovers_exact_cubic()
    test_higher_degree_fits_curve_better_than_a_line()
    print("all tests passed")
