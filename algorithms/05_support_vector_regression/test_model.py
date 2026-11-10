import numpy as np
from model import LinearSVR


def _line_data(noise=0.05):
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 1))
    y = 1.5 * X[:, 0] + 0.5 + rng.normal(scale=noise, size=200)
    return X, y


def test_recovers_a_line():
    X, y = _line_data()
    model = LinearSVR(epsilon=0.05, lam=1e-4, n_iters=3000).fit(X, y)
    assert abs(model.weights[0] - 1.5) < 0.1
    assert abs(model.bias - 0.5) < 0.1


def test_robust_to_outliers_unlike_least_squares():
    X, y = _line_data()
    y_bad = y.copy()
    y_bad[:5] += 50.0  # a few huge outliers
    svr = LinearSVR(epsilon=0.05, lam=1e-4, n_iters=3000).fit(X, y_bad)
    ols_slope = np.polyfit(X[:, 0], y_bad, 1)[0]
    # the linear (not squared) loss limits each outlier's pull
    assert abs(svr.weights[0] - 1.5) < abs(ols_slope - 1.5)


def test_wide_tube_ignores_small_noise():
    # every point sits inside a tube of half-width 5 around the mean: no gradient, no slope learned
    X, y = _line_data()
    model = LinearSVR(epsilon=5.0, lam=1e-4, n_iters=500).fit(X, y)
    assert abs(model.weights[0]) < 1e-9


if __name__ == "__main__":
    test_recovers_a_line()
    test_robust_to_outliers_unlike_least_squares()
    test_wide_tube_ignores_small_noise()
    print("all tests passed")
