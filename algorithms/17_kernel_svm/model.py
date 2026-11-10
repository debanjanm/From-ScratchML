import numpy as np


def _kernel(A, B, kind, gamma, degree):
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (gamma * (A @ B.T) + 1.0) ** degree
    if kind == "rbf":
        sq_dist = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * sq_dist)
    raise ValueError(f"unknown kernel: {kind}")


class KernelSVM:
    """Binary SVM in the dual form, trained with simplified SMO (Sequential Minimal Optimization).

    The kernel lets the SVM draw curved boundaries: it computes dot products in a high-dimensional
    feature space without ever building that space. Prediction only needs the support vectors:

        f(x) = sum_i alpha_i * y_i * k(x_i, x) + b

    SMO repeatedly picks two multipliers (alpha_i, alpha_j), solves that tiny 2-variable problem
    exactly, and moves on; sum(alpha * y) = 0 and 0 <= alpha <= C are kept throughout.

    ponytail: the partner j is picked at random (not by max |Ei - Ej|) and the full n x n kernel
    matrix is stored. Fine for a few thousand samples; swap in working-set selection for more."""

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, degree=3, tol=1e-3, max_passes=5,
                 max_iter=200, random_state=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_passes = max_passes  # stop after this many full sweeps with no change
        self.max_iter = max_iter      # hard cap on sweeps, in case it never settles
        self.random_state = random_state
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("KernelSVM is binary: needs exactly 2 classes")
        y_pm = 2.0 * y_enc - 1.0

        n = len(X)
        K = _kernel(X, X, self.kernel, self.gamma, self.degree)
        rng = np.random.default_rng(self.random_state)
        alpha, b = np.zeros(n), 0.0
        C, tol = self.C, self.tol

        passes = sweeps = 0
        while passes < self.max_passes and sweeps < self.max_iter:
            changed = 0
            for i in range(n):
                E_i = (alpha * y_pm) @ K[:, i] + b - y_pm[i]  # prediction error on sample i
                # violates the KKT conditions? (inside the margin with room to move)
                if (y_pm[i] * E_i < -tol and alpha[i] < C) or (y_pm[i] * E_i > tol and alpha[i] > 0):
                    j = rng.choice(np.delete(np.arange(n), i))
                    E_j = (alpha * y_pm) @ K[:, j] + b - y_pm[j]
                    a_i_old, a_j_old = alpha[i], alpha[j]

                    # box that keeps alpha_j inside [0, C] while sum(alpha*y) stays fixed
                    if y_pm[i] != y_pm[j]:
                        L, H = max(0.0, a_j_old - a_i_old), min(C, C + a_j_old - a_i_old)
                    else:
                        L, H = max(0.0, a_i_old + a_j_old - C), min(C, a_i_old + a_j_old)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]  # curvature along the constraint line
                    if L == H or eta >= 0:
                        continue

                    a_j_new = np.clip(a_j_old - y_pm[j] * (E_i - E_j) / eta, L, H)
                    if abs(a_j_new - a_j_old) < 1e-5:
                        continue  # too small to matter; leave BOTH alphas untouched
                    alpha[j] = a_j_new
                    alpha[i] = a_i_old + y_pm[i] * y_pm[j] * (a_j_old - a_j_new)

                    b1 = b - E_i - y_pm[i] * (alpha[i] - a_i_old) * K[i, i] - y_pm[j] * (alpha[j] - a_j_old) * K[i, j]
                    b2 = b - E_j - y_pm[i] * (alpha[i] - a_i_old) * K[i, j] - y_pm[j] * (alpha[j] - a_j_old) * K[j, j]
                    b = b1 if 0 < alpha[i] < C else b2 if 0 < alpha[j] < C else (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            sweeps += 1

        sv = alpha > 1e-8  # support vectors: the only points with non-zero alpha
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (alpha * y_pm)[sv]
        self.bias = b
        self.alpha_ = alpha
        return self

    def decision_function(self, X):
        K = _kernel(np.asarray(X, dtype=float), self.support_vectors_, self.kernel, self.gamma, self.degree)
        return K @ self.dual_coef_ + self.bias

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
