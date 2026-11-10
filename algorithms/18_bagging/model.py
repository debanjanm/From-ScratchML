import numpy as np


class _Tree:
    """Plain CART classifier (Gini, all features). The default base learner for Bagging."""

    def __init__(self, max_depth=None, min_samples_split=2, n_classes=2):
        self.max_depth = max_depth if max_depth is not None else 10**9
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes

    def fit(self, X, y):
        self.root = self._grow(X, np.eye(self.n_classes)[y], 0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _grow(self, X, Y, depth):
        counts = Y.sum(axis=0)
        n = len(Y)
        if depth >= self.max_depth or n < self.min_samples_split or counts.max() == n:
            return counts.argmax()

        best_score, best_feature, best_threshold = (counts**2).sum() / n + 1e-12, None, None
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f])
            xs = X[order, f]
            left = np.cumsum(Y[order], axis=0)[:-1]
            n_left = np.arange(1, n)[:, None]
            right = counts - left
            # maximizing sum(c^2/n) over both sides == minimizing the weighted Gini impurity
            score = (left**2 / n_left).sum(axis=1) + (right**2 / (n - n_left)).sum(axis=1)
            score = np.where(xs[1:] != xs[:-1], score, -np.inf)
            i = score.argmax()
            if score[i] > best_score:
                best_score, best_feature, best_threshold = score[i], f, (xs[i] + xs[i + 1]) / 2

        if best_feature is None:
            return counts.argmax()
        mask = X[:, best_feature] <= best_threshold
        return (best_feature, best_threshold,
                self._grow(X[mask], Y[mask], depth + 1), self._grow(X[~mask], Y[~mask], depth + 1))

    def _traverse(self, x, node):
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if x[feature] <= threshold else right
        return node


class BaggingClassifier:
    """Bootstrap AGGregatING: train each model on a random resample (with replacement) of the
    data, then majority-vote. Averaging many high-variance models cuts the variance while the
    bias stays the same -- which is why it helps unstable learners like deep trees.

    make_estimator: zero-argument function returning a NEW unfitted model with fit(X, y) and
    predict(X). Default is a CART tree, but any learner works (e.g. lambda: KNN(k=3)).
    Labels passed to the base models are integers 0..k-1.

    After fitting, oob_score_ is accuracy measured on the samples each model did NOT see:
    a free validation estimate, no hold-out set needed."""

    def __init__(self, make_estimator=None, n_estimators=25, max_samples=1.0, max_depth=None,
                 random_state=0):
        self.make_estimator = make_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        n, k = len(X), len(self.classes_)
        make = self.make_estimator or (lambda: _Tree(self.max_depth, n_classes=k))

        rng = np.random.default_rng(self.random_state)
        size = max(1, int(self.max_samples * n))
        self.estimators_ = []
        oob_votes = np.zeros((n, k))
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size)  # bootstrap sample
            model = make().fit(X[idx], y_enc[idx])
            self.estimators_.append(model)
            out_of_bag = np.setdiff1d(np.arange(n), idx)
            if len(out_of_bag):
                oob_votes[out_of_bag, model.predict(X[out_of_bag])] += 1

        seen = oob_votes.sum(axis=1) > 0  # a sample that was in every bootstrap has no OOB vote
        self.oob_score_ = (oob_votes[seen].argmax(axis=1) == y_enc[seen]).mean()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.array([m.predict(X) for m in self.estimators_])  # (n_estimators, n_samples)
        k = len(self.classes_)
        winners = [np.bincount(col, minlength=k).argmax() for col in votes.T]
        return self.classes_[winners]
