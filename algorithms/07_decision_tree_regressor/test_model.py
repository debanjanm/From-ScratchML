import numpy as np
from model import DecisionTreeRegressor


def test_fits_step_function_exactly():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([5.0, 5.0, 5.0, 20.0, 20.0, 20.0])
    assert np.allclose(DecisionTreeRegressor(max_depth=1).fit(X, y).predict(X), y)


def test_depth_zero_predicts_the_mean():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0)
    assert np.allclose(DecisionTreeRegressor(max_depth=0).fit(X, y).predict(X), y.mean())


def test_predictions_are_piecewise_constant_and_deeper_fits_better():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 6, size=(300, 1))
    y = np.sin(X[:, 0]) + rng.normal(scale=0.05, size=300)
    shallow = DecisionTreeRegressor(max_depth=2).fit(X, y)
    deep = DecisionTreeRegressor(max_depth=6).fit(X, y)
    assert len(np.unique(shallow.predict(X))) <= 4  # 2**depth leaves at most
    assert np.mean((deep.predict(X) - y) ** 2) < np.mean((shallow.predict(X) - y) ** 2)


if __name__ == "__main__":
    test_fits_step_function_exactly()
    test_depth_zero_predicts_the_mean()
    test_predictions_are_piecewise_constant_and_deeper_fits_better()
    print("all tests passed")
