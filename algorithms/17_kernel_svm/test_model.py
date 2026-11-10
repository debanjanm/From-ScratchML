import numpy as np
from model import KernelSVM


def _circles(seed=0, n=200):
    # inner disc vs outer ring: impossible for any straight line
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(np.arange(n) < n // 2, 1.0, 3.0) + rng.normal(scale=0.15, size=n)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = np.where(np.arange(n) < n // 2, "inner", "outer")
    return X, y


def test_rbf_solves_circles_but_linear_kernel_cannot():
    X, y = _circles()
    rbf = KernelSVM(kernel="rbf", gamma=0.5, C=10.0).fit(X, y)
    linear = KernelSVM(kernel="linear", C=10.0).fit(X, y)
    assert (rbf.predict(X) == y).mean() > 0.97
    assert (linear.predict(X) == y).mean() < 0.75


def test_dual_constraints_hold():
    X, y = _circles(seed=1)
    model = KernelSVM(kernel="rbf", gamma=0.5, C=5.0).fit(X, y)
    assert (model.alpha_ >= -1e-9).all() and (model.alpha_ <= 5.0 + 1e-9).all()  # 0 <= alpha <= C
    assert abs(model.dual_coef_.sum()) < 1e-6                                     # sum(alpha * y) = 0


def test_only_support_vectors_are_kept():
    X, y = _circles(seed=2)
    model = KernelSVM(kernel="rbf", gamma=0.5, C=10.0).fit(X, y)
    assert 0 < len(model.support_vectors_) < len(X)


def test_rejects_multiclass():
    try:
        KernelSVM().fit(np.zeros((3, 2)), np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_rbf_solves_circles_but_linear_kernel_cannot()
    test_dual_constraints_hold()
    test_only_support_vectors_are_kept()
    test_rejects_multiclass()
    print("all tests passed")
