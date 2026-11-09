import numpy as np
from model import LassoRegression, RidgeRegression


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    true_w = np.array([4.0, -3.0, 2.0, 0.0, 0.0, 0.0])  # last three features are irrelevant
    y = X @ true_w + 5.0 + rng.normal(scale=0.5, size=200)
    return X, y, true_w


def test_ridge_alpha_zero_is_ordinary_least_squares():
    X, y, true_w = _data()
    model = RidgeRegression(alpha=0.0).fit(X, y)
    assert np.allclose(model.weights, true_w, atol=0.2)
    assert abs(model.bias - 5.0) < 0.2


def test_ridge_shrinks_weights_as_alpha_grows():
    X, y, _ = _data()
    norms = [np.linalg.norm(RidgeRegression(alpha=a).fit(X, y).weights) for a in (0, 10, 1000, 1e6)]
    assert norms == sorted(norms, reverse=True)
    assert norms[-1] < 0.01


def test_lasso_zeroes_irrelevant_features():
    X, y, true_w = _data()
    model = LassoRegression(alpha=0.1).fit(X, y)
    assert (model.weights[3:] == 0).all()  # exact zeros, not just small
    assert np.allclose(model.weights[:3], true_w[:3], atol=0.3)


def test_lasso_huge_alpha_gives_all_zero_weights():
    X, y, _ = _data()
    model = LassoRegression(alpha=1e3).fit(X, y)
    assert (model.weights == 0).all()
    assert abs(model.bias - y.mean()) < 1e-9


if __name__ == "__main__":
    test_ridge_alpha_zero_is_ordinary_least_squares()
    test_ridge_shrinks_weights_as_alpha_grows()
    test_lasso_zeroes_irrelevant_features()
    test_lasso_huge_alpha_gives_all_zero_weights()
    print("all tests passed")
