import numpy as np
from model import LDAProjection


def _two_stretched_blobs(seed=0, n=200):
    """Huge variance along x (useless), classes differ only along y (useful)."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0, 0], [10.0, 0.5], size=(n, 2)), rng.normal([0, 3], [10.0, 0.5], size=(n, 2))])
    return X, np.array([0] * n + [1] * n)


def _fisher_ratio(z, y):
    between = (z[y == 0].mean() - z[y == 1].mean()) ** 2
    within = z[y == 0].var() + z[y == 1].var()
    return between / within


def test_finds_the_discriminative_axis_where_pca_picks_the_noisy_one():
    X, y = _two_stretched_blobs()
    lda = LDAProjection(n_components=1).fit_transform(X, y)[:, 0]
    # PCA's first direction = largest variance = the x axis
    Xc = X - X.mean(axis=0)
    pca = Xc @ np.linalg.eigh(np.cov(Xc, rowvar=False))[1][:, -1]
    assert _fisher_ratio(lda, y) > 10 * _fisher_ratio(pca, y)
    assert abs(LDAProjection(n_components=1).fit(X, y).components_[0][1]) > 0.99 * np.linalg.norm(
        LDAProjection(n_components=1).fit(X, y).components_[0]
    )  # points along y


def test_at_most_classes_minus_one_directions():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 5)) + np.repeat(np.eye(3, 5) * 4, 30, axis=0)
    y = np.repeat([0, 1, 2], 30)
    assert LDAProjection().fit_transform(X, y).shape == (90, 2)
    try:
        LDAProjection(n_components=3).fit(X, y)
        raise AssertionError("should reject 3 components for 3 classes")
    except ValueError:
        pass


def test_directions_are_whitened_and_uncorrelated_within_class():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 4)) @ rng.normal(size=(4, 4)) + np.repeat(rng.normal(size=(3, 4)) * 5, 50, axis=0)
    y = np.repeat([0, 1, 2], 50)
    model = LDAProjection().fit(X, y)
    Z = model.transform(X)
    Sw = sum((Z[y == c] - Z[y == c].mean(0)).T @ (Z[y == c] - Z[y == c].mean(0)) for c in range(3))
    assert np.allclose(Sw, np.eye(2), atol=1e-4)   # w' Sw w = 1 and cross terms 0
    assert (np.diff(model.eigenvalues_) <= 1e-12).all()


if __name__ == "__main__":
    test_finds_the_discriminative_axis_where_pca_picks_the_noisy_one()
    test_at_most_classes_minus_one_directions()
    test_directions_are_whitened_and_uncorrelated_within_class()
    print("all tests passed")
