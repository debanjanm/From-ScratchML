import numpy as np
from model import BaggingClassifier, _Tree


def _noisy_data(seed, n=400, flip=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    noisy = rng.random(n) < flip
    return X, np.where(noisy, 1 - y, y), (X[:, 0] + X[:, 1] > 0).astype(int)  # last = true labels


def test_tree_learns_a_simple_split():
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert (_Tree(n_classes=2).fit(X, y).predict(X) == y).all()


def test_bagging_beats_one_overfit_tree_on_noisy_labels():
    X, y, _ = _noisy_data(0)
    X_test, _, y_test_true = _noisy_data(1, n=1000)
    single = _Tree(n_classes=2).fit(X, y)  # unlimited depth: memorizes the label noise
    bag = BaggingClassifier(n_estimators=40).fit(X, y)
    acc = lambda m: (m.predict(X_test) == y_test_true).mean()
    assert acc(bag) > acc(single) + 0.02


def test_oob_score_estimates_generalization_without_a_holdout():
    X, y, _ = _noisy_data(2)
    X_test, y_test, _ = _noisy_data(3, n=2000)
    model = BaggingClassifier(n_estimators=60).fit(X, y)
    assert abs(model.oob_score_ - (model.predict(X_test) == y_test).mean()) < 0.06


def test_any_base_learner_and_string_labels():
    class Centroid:  # tiny base learner: nearest class mean
        def fit(self, X, y):
            self.m = np.array([X[y == c].mean(axis=0) for c in np.unique(y)])
            self.c = np.unique(y)
            return self

        def predict(self, X):
            return self.c[np.argmin(((X[:, None] - self.m[None]) ** 2).sum(-1), axis=1)]

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 1, (50, 2)), rng.normal(3, 1, (50, 2))])
    y = np.array(["no"] * 50 + ["yes"] * 50)
    model = BaggingClassifier(make_estimator=Centroid, n_estimators=10).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.97


if __name__ == "__main__":
    test_tree_learns_a_simple_split()
    test_bagging_beats_one_overfit_tree_on_noisy_labels()
    test_oob_score_estimates_generalization_without_a_holdout()
    test_any_base_learner_and_string_labels()
    print("all tests passed")
