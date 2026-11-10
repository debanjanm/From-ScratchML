import numpy as np
from model import XGBoost


def _tiny():
    return np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 1.0, 3.0, 3.0])


def test_leaf_weight_matches_the_formula():
    # base score = mean = 2, so g = pred - y = [1, 1, -1, -1], h = 1.
    # Left leaf: G = 2, H = 2  ->  w = -G / (H + lambda) = -2/3 for lambda = 1.
    X, y = _tiny()
    model = XGBoost(n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=1.0).fit(X, y)
    assert np.allclose(model.predict(X), [2 - 2 / 3, 2 - 2 / 3, 2 + 2 / 3, 2 + 2 / 3])


def test_lambda_shrinks_the_update():
    X, y = _tiny()
    small = XGBoost(n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=0.0).fit(X, y).predict(X)
    large = XGBoost(n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=100.0).fit(X, y).predict(X)
    assert np.allclose(small, y)                      # lambda = 0: leaves hit the exact mean
    assert np.abs(large - 2.0).max() < 0.05           # heavy L2: barely moves from the base score


def test_gamma_prunes_unprofitable_splits():
    X, y = _tiny()
    model = XGBoost(n_estimators=5, learning_rate=1.0, max_depth=3, gamma=100.0).fit(X, y)
    assert np.allclose(model.predict(X), 2.0)         # no split worth gamma: the model stays constant


def test_binary_logistic():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, "pos", "neg")  # XOR pattern
    model = XGBoost(objective="binary:logistic", n_estimators=50, max_depth=3).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0) and ((probs > 0) & (probs < 1)).all()


if __name__ == "__main__":
    test_leaf_weight_matches_the_formula()
    test_lambda_shrinks_the_update()
    test_gamma_prunes_unprofitable_splits()
    test_binary_logistic()
    print("all tests passed")
