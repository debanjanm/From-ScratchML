import numpy as np
from model import KernelPCA


def test_training_scores_are_sqrt_eigenvalue_times_eigenvector():
    # transform() on the training set must agree with the textbook scores sqrt(lambda) * v
    X = np.random.default_rng(0).normal(size=(40, 3))
    model = KernelPCA(n_components=3, kernel="rbf", gamma=0.5).fit(X)
    scores = model.transform(X)
    assert np.allclose((scores ** 2).sum(axis=0), model.eigenvalues_)  # ||score||^2 = lambda
    assert np.allclose(scores.mean(axis=0), 0.0, atol=1e-10)             # centered in feature space


def test_linear_kernel_equals_pca():
    X = np.random.default_rng(1).normal(size=(60, 4)) * [5, 3, 1, 0.5]
    Z = KernelPCA(n_components=2, kernel="linear").fit_transform(X)
    Xc = X - X.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(Xc, rowvar=False))
    pca = Xc @ vecs[:, ::-1][:, :2]
    assert np.allclose(np.abs(Z), np.abs(pca), atol=1e-8)


def test_rbf_unrolls_concentric_circles_where_pca_cannot():
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=200, factor=0.3, noise=0.04, random_state=0)

    def separable(z):  # can one threshold on this axis split the two rings?
        a, b = z[y == 0], z[y == 1]
        return a.max() < b.min() or b.max() < a.min()

    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=10).fit_transform(X)
    assert any(separable(kpca[:, j]) for j in range(2))
    linear = KernelPCA(n_components=2, kernel="linear").fit_transform(X)
    assert not any(separable(linear[:, j]) for j in range(2))


def test_transform_new_points_matches_training_rows():
    X = np.random.default_rng(2).normal(size=(30, 2))
    model = KernelPCA(n_components=2, kernel="poly", gamma=1.0, degree=2).fit(X)
    assert model.transform(X[:5]).shape == (5, 2)
    assert np.allclose(model.transform(X[:5]), model.fit_transform(X)[:5])


if __name__ == "__main__":
    test_training_scores_are_sqrt_eigenvalue_times_eigenvector()
    test_linear_kernel_equals_pca()
    test_rbf_unrolls_concentric_circles_where_pca_cannot()
    test_transform_new_points_matches_training_rows()
    print("all tests passed")
