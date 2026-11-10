import numpy as np
from model import PCA


def test_reduces_dimensionality():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    model = PCA(n_components=2).fit(X)
    transformed = model.transform(X)
    assert transformed.shape == (100, 2)


def test_recovers_dominant_axis():
    rng = np.random.default_rng(0)
    # variance almost entirely along the x-axis
    X = np.column_stack([rng.normal(scale=10, size=500), rng.normal(scale=0.1, size=500)])
    model = PCA(n_components=1).fit(X)
    assert model.explained_variance_ratio_[0] > 0.99
    # first component should point mostly along the x-axis (sign is arbitrary)
    assert abs(model.components[0][0]) > 0.99


def test_explained_variance_ratios_sum_to_one_when_full_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    model = PCA(n_components=4).fit(X)
    assert abs(model.explained_variance_ratio_.sum() - 1.0) < 1e-9


if __name__ == "__main__":
    test_reduces_dimensionality()
    test_recovers_dominant_axis()
    test_explained_variance_ratios_sum_to_one_when_full_rank()
    print("all tests passed")
