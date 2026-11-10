import numpy as np
from model import GaussianProcessRegressor


def test_interpolates_training_points_when_noise_is_tiny():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 0.0, -1.0])
    gp = GaussianProcessRegressor(length_scale=1.0, noise=1e-4).fit(X, y)
    assert np.allclose(gp.predict(X), y, atol=1e-3)


def test_uncertainty_is_small_near_data_and_large_far_away():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.sin(X[:, 0])
    gp = GaussianProcessRegressor(length_scale=1.0, variance=1.0, noise=0.05).fit(X, y)
    _, std_near = gp.predict(np.array([[1.0]]), return_std=True)
    _, std_far = gp.predict(np.array([[50.0]]), return_std=True)
    assert std_near[0] < 0.1
    assert abs(std_far[0] - 1.0) < 1e-6  # reverts to the prior std = sqrt(variance)


def test_far_from_data_mean_reverts_to_zero_prior():
    X = np.array([[0.0], [1.0]])
    gp = GaussianProcessRegressor().fit(X, np.array([5.0, 5.0]))
    assert abs(gp.predict(np.array([[100.0]]))[0]) < 1e-6


if __name__ == "__main__":
    test_interpolates_training_points_when_noise_is_tiny()
    test_uncertainty_is_small_near_data_and_large_far_away()
    test_far_from_data_mean_reverts_to_zero_prior()
    print("all tests passed")
