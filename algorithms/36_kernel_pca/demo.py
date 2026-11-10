import numpy as np
from model import KernelPCA
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA as SklearnKPCA, PCA

X, y = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=0)
X_new, _ = make_circles(n_samples=50, factor=0.3, noise=0.05, random_state=7)

for name, params in {"rbf": dict(kernel="rbf", gamma=10), "poly": dict(kernel="poly", gamma=1.0, degree=2), "linear": dict(kernel="linear")}.items():
    ours = KernelPCA(n_components=2, **params).fit(X)
    theirs = SklearnKPCA(n_components=2, **params).fit(X)
    # eigenvector sign is arbitrary, so compare magnitudes
    assert np.allclose(np.abs(ours.transform(X)), np.abs(theirs.transform(X)), atol=1e-6), name
    assert np.allclose(np.abs(ours.transform(X_new)), np.abs(theirs.transform(X_new)), atol=1e-6), name
    print("%-6s matches sklearn (training points and unseen points)" % name)


def best_threshold_accuracy(z):  # best single-threshold split of the two rings, either orientation
    acc = np.array([((z > t) == y).mean() for t in z])
    return np.maximum(acc, 1 - acc).max()


best = lambda Z: max(best_threshold_accuracy(Z[:, j]) for j in range(2))
rbf = KernelPCA(n_components=2, kernel="rbf", gamma=10).fit_transform(X)
pca = PCA(n_components=2).fit_transform(X)
print("best 1-D threshold accuracy on the rings: kernel PCA=%.2f  PCA=%.2f" % (best(rbf), best(pca)))
assert best(rbf) > 0.99 and best(pca) < 0.8
