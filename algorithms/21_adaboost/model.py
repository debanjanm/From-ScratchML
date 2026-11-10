import numpy as np


class _Stump:
    """One-split tree: predicts `polarity` when x[feature] > threshold, else -polarity."""

    def fit(self, X, y, w):
        best_err = np.inf
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f])
            xs, ys, ws = X[order, f], y[order], w[order]
            # weighted error of "left = -1, right = +1" for every split point, via cumulative sums
            pos_left = np.cumsum(ws * (ys == 1))[:-1]
            neg_left = np.cumsum(ws * (ys == -1))[:-1]
            err = pos_left + (np.sum(ws * (ys == -1)) - neg_left)
            err = np.where(xs[1:] != xs[:-1], err, 0.5)  # can't split between equal values
            for polarity, e in ((1, err), (-1, 1 - err)):  # flipping the stump flips its error
                i = e.argmin()
                if e[i] < best_err:
                    best_err = e[i]
                    self.feature, self.polarity = f, polarity
                    self.threshold = (xs[i] + xs[i + 1]) / 2
        return self

    def predict(self, X):
        return self.polarity * np.where(X[:, self.feature] > self.threshold, 1, -1)


class AdaBoostClassifier:
    """Binary AdaBoost with decision stumps. Each round up-weights the samples the previous
    stumps got wrong; the final prediction is a weighted vote of all stumps."""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.classes_ = None
        self.stumps, self.alphas = [], []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier is binary: needs exactly 2 classes")
        y_pm = 2 * y_enc - 1  # {0, 1} -> {-1, +1}

        w = np.full(len(y_pm), 1 / len(y_pm))
        self.stumps, self.alphas = [], []
        for _ in range(self.n_estimators):
            stump = _Stump().fit(X, y_pm, w)
            pred = stump.predict(X)
            err = np.clip(w[pred != y_pm].sum(), 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)  # accurate stumps get a bigger say
            w *= np.exp(-alpha * y_pm * pred)  # wrong samples grow, right ones shrink
            w /= w.sum()
            self.stumps.append(stump)
            self.alphas.append(alpha)
            if err <= 1e-10:  # perfect stump: nothing left to fix
                break
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
