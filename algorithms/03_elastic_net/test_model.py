import numpy as np
from model import ElasticNet


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    true_w = np.array([4.0, -3.0, 2.0, 0.0, 0.0, 0.0])
    y = X @ true_w + 5.0 + rng.normal(scale=0.5, size=200)
    return X, y, true_w


def test_l1_ratio_one_zeroes_irrelevant_features_like_lasso():
    X, y, true_w = _data()
    model = ElasticNet(alpha=0.1, l1_ratio=1.0).fit(X, y)
    assert (model.weights[3:] == 0).all()
    assert np.allclose(model.weights[:3], true_w[:3], atol=0.3)


def test_l1_ratio_zero_matches_ridge_closed_form():
    X, y, _ = _data()
    n, alpha = len(X), 0.3
    model = ElasticNet(alpha=alpha, l1_ratio=0.0).fit(X, y)

    Xc, yc = X - X.mean(axis=0), y - y.mean()
    ridge_w = np.linalg.solve(Xc.T @ Xc + n * alpha * np.eye(6), Xc.T @ yc)  # same objective, ridge form
    assert np.allclose(model.weights, ridge_w, atol=1e-6)


def test_keeps_correlated_features_together():
    # two near-copies of the same signal: pure L1 tends to pick one, the L2 part shares the weight
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)
    X = np.column_stack([base, base + rng.normal(scale=0.01, size=300), rng.normal(size=300)])
    y = 3 * base + rng.normal(scale=0.1, size=300)
    w = ElasticNet(alpha=0.05, l1_ratio=0.3).fit(X, y).weights
    assert w[0] > 0.5 and w[1] > 0.5
    assert abs(w[0] - w[1]) < 0.3


if __name__ == "__main__":
    test_l1_ratio_one_zeroes_irrelevant_features_like_lasso()
    test_l1_ratio_zero_matches_ridge_closed_form()
    test_keeps_correlated_features_together()
    print("all tests passed")
