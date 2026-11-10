import numpy as np
from model import TruncatedSVD


def test_singular_values_match_numpy_svd():
    X = np.random.default_rng(0).normal(size=(60, 6))
    model = TruncatedSVD(n_components=6).fit(X)
    assert np.allclose(model.singular_values_, np.linalg.svd(X, compute_uv=False), atol=1e-8)


def test_reconstruction_error_is_sum_of_discarded_squared_singular_values():
    # Eckart-Young: ||X - X_k||_F^2 = sum of the squared singular values we dropped
    X = np.random.default_rng(1).normal(size=(80, 7))
    s = np.linalg.svd(X, compute_uv=False)
    model = TruncatedSVD(n_components=3).fit(X)
    error = ((X - model.inverse_transform(model.transform(X))) ** 2).sum()
    assert np.isclose(error, (s[3:] ** 2).sum())


def test_exact_when_data_is_low_rank():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 8))  # rank 2
    model = TruncatedSVD(n_components=2).fit(X)
    assert np.allclose(model.inverse_transform(model.transform(X)), X, atol=1e-8)


def test_components_orthonormal_and_data_not_centered():
    X = np.random.default_rng(3).normal(size=(40, 5)) + 100.0   # huge offset
    model = TruncatedSVD(n_components=3).fit(X)
    assert np.allclose(model.components_ @ model.components_.T, np.eye(3), atol=1e-8)
    # the offset dominates, so component 0 points along the mean direction (what PCA would remove)
    mean_dir = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    assert abs(model.components_[0] @ mean_dir) > 0.99


if __name__ == "__main__":
    test_singular_values_match_numpy_svd()
    test_reconstruction_error_is_sum_of_discarded_squared_singular_values()
    test_exact_when_data_is_low_rank()
    test_components_orthonormal_and_data_not_centered()
    print("all tests passed")
