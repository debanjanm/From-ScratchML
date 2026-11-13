import numpy as np
from model import BayesianLinearRegression


def _data(seed=0, n=200, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    w = np.array([2.0, -1.0, 0.0, 0.5])
    return X, X @ w + 3.0 + rng.normal(scale=noise, size=n), w


def test_posterior_matches_textbook_formulas_and_ridge():
    X, y, _ = _data()
    model = BayesianLinearRegression().fit(X, y)
    Xc, yc = X - X.mean(0), y - y.mean()
    a, l = model.alpha_, model.lambda_
    S = np.linalg.inv(l * np.eye(4) + a * Xc.T @ Xc)
    assert np.allclose(model.sigma_, S)
    assert np.allclose(model.coef_, a * S @ Xc.T @ yc)
    ridge = np.linalg.solve(Xc.T @ Xc + (l / a) * np.eye(4), Xc.T @ yc)  # posterior mean == ridge with strength lambda/alpha
    assert np.allclose(model.coef_, ridge)


def test_learns_noise_level_and_weights():
    X, y, w = _data(n=2000, noise=0.5)
    model = BayesianLinearRegression().fit(X, y)
    assert abs(1 / np.sqrt(model.alpha_) - 0.5) < 0.03       # recovered noise std
    assert np.allclose(model.coef_, w, atol=0.05) and abs(model.intercept_ - 3.0) < 0.05


def test_uncertainty_grows_away_from_the_data():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(30, 1))
    y = 2 * x[:, 0] + rng.normal(scale=0.2, size=30)
    model = BayesianLinearRegression().fit(x, y)
    _, std = model.predict([[0.0], [1.0], [10.0], [100.0]], return_std=True)
    assert std[0] < std[1] < std[2] < std[3]
    assert std[0] > 1 / np.sqrt(model.alpha_) - 1e-9           # never below the noise floor


def test_interval_coverage_is_calibrated():
    X, y, _ = _data(seed=2, n=300)
    Xt, yt, _ = _data(seed=3, n=3000)
    mean, std = BayesianLinearRegression().fit(X, y).predict(Xt, return_std=True)
    coverage = (np.abs(yt - mean) < 1.96 * std).mean()
    assert abs(coverage - 0.95) < 0.02


def test_irrelevant_features_are_shrunk_hard_with_few_samples():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 30))                              # 30 features, only the first matters
    y = 3 * X[:, 0] + rng.normal(scale=0.3, size=40)
    coef = BayesianLinearRegression().fit(X, y).coef_
    assert abs(coef[0] - 3) < 0.3 and np.abs(coef[1:]).max() < 0.3


if __name__ == "__main__":
    test_posterior_matches_textbook_formulas_and_ridge()
    test_learns_noise_level_and_weights()
    test_uncertainty_grows_away_from_the_data()
    test_interval_coverage_is_calibrated()
    test_irrelevant_features_are_shrunk_hard_with_few_samples()
    print("all tests passed")
