import numpy as np
from model import StackingClassifier, VotingClassifier


class Fixed:
    """Stub model: ignores the data and always answers the same."""

    def __init__(self, label=None, proba=None):
        self.label, self.proba = label, proba

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.label)

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


class Centroid:
    """Small real learner: nearest class mean, with probabilities from distances."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        d = ((X[:, None] - self.means[None]) ** 2).sum(-1)
        e = np.exp(-d + d.min(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2, 1, (60, 2)), rng.normal(2, 1, (60, 2))])
    return X, np.array([0] * 60 + [1] * 60)


def test_hard_voting_is_majority():
    X = np.zeros((4, 1))
    votes = [("a", Fixed(1)), ("b", Fixed(1)), ("c", Fixed(0))]
    model = VotingClassifier(votes, voting="hard").fit(X, np.array([0, 1, 0, 1]))
    assert (model.predict(X) == 1).all()


def test_soft_voting_can_overrule_the_majority():
    # two models mildly prefer class 1, one is very sure about class 0: hard says 1, soft says 0
    X = np.zeros((3, 1))
    y = np.array([0, 1, 0])
    stubs = [("a", Fixed(0, [0.9, 0.1])), ("b", Fixed(1, [0.4, 0.6])), ("c", Fixed(1, [0.4, 0.6]))]
    assert (VotingClassifier(stubs, voting="hard").fit(X, y).predict(X) == 1).all()
    assert (VotingClassifier(stubs, voting="soft").fit(X, y).predict(X) == 0).all()


def test_stacking_uses_out_of_fold_data():
    class Spy(Centroid):
        sizes = []

        def fit(self, X, y):
            Spy.sizes.append(len(X))
            return super().fit(X, y)

    X, y = _blobs()
    StackingClassifier([("spy", Spy())], cv=5).fit(X, y)
    # five fits on 80% of the data (the out-of-fold ones), then one refit on everything
    assert Spy.sizes == [96] * 5 + [120]


def test_stacking_learns_to_trust_the_good_model():
    X, y = _blobs(1)
    model = StackingClassifier([("good", Centroid()), ("useless", Fixed(0, [0.5, 0.5]))]).fit(X, y)
    X_new, y_new = _blobs(2)
    assert (model.predict(X_new) == y_new).mean() > 0.9


if __name__ == "__main__":
    test_hard_voting_is_majority()
    test_soft_voting_can_overrule_the_majority()
    test_stacking_uses_out_of_fold_data()
    test_stacking_learns_to_trust_the_good_model()
    print("all tests passed")
