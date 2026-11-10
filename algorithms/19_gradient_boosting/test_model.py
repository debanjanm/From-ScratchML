import numpy as np
from model import GradientBoostingRegressor, _RegressionTree


def test_tree_splits_step_function():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([5.0, 5.0, 5.0, 20.0, 20.0, 20.0])
    preds = _RegressionTree(max_depth=1).fit(X, y).predict(X)
    assert np.allclose(preds, y)


def test_fits_sine_and_more_trees_help():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 6, size=(300, 1))
    y = np.sin(X[:, 0]) + rng.normal(scale=0.05, size=300)

    mse = lambda m: np.mean((m.predict(X) - y) ** 2)
    few = GradientBoostingRegressor(n_estimators=5).fit(X, y)
    many = GradientBoostingRegressor(n_estimators=100).fit(X, y)
    assert mse(many) < mse(few)
    assert mse(many) < 0.01


def test_zero_trees_predicts_mean():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0)
    model = GradientBoostingRegressor(n_estimators=0).fit(X, y)
    assert np.allclose(model.predict(X), y.mean())


if __name__ == "__main__":
    test_tree_splits_step_function()
    test_fits_sine_and_more_trees_help()
    test_zero_trees_predicts_mean()
    print("all tests passed")
