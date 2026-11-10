import copy

import numpy as np

# These ensembles COMBINE other models, so the base learners are plugged in from outside: any
# object with fit(X, y) and predict(X), and optionally predict_proba(X) (columns in sorted-class
# order, like scikit-learn). Pass models from this repo or from anywhere else.


def _probabilities(model, X, classes):
    """(n_samples, n_classes) probabilities; a model without predict_proba casts a one-hot vote."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    return (np.asarray(model.predict(X))[:, None] == classes[None, :]).astype(float)


class VotingClassifier:
    """Train several different models on the same data and let them vote.

    voting="hard": each model casts one vote for its predicted class; majority wins.
    voting="soft": average the models' predicted probabilities; highest average wins. Soft voting
                   uses each model's confidence, so one very sure model can outweigh two unsure ones."""

    def __init__(self, estimators, voting="hard"):
        self.estimators = estimators  # list of (name, unfitted_model)
        self.voting = voting

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.models_ = [copy.deepcopy(est).fit(X, y) for _, est in self.estimators]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.voting == "soft":
            avg = np.mean([_probabilities(m, X, self.classes_) for m in self.models_], axis=0)
            return self.classes_[avg.argmax(axis=1)]
        # hard: count how many models chose each class
        picks = np.array([np.searchsorted(self.classes_, m.predict(X)) for m in self.models_])
        counts = np.array([np.bincount(col, minlength=len(self.classes_)) for col in picks.T])
        return self.classes_[counts.argmax(axis=1)]


class _Softmax:
    """Default meta-learner: multiclass logistic regression by gradient descent."""

    def __init__(self, lr=0.5, n_iters=1000):
        self.lr, self.n_iters = lr, n_iters

    def fit(self, X, y):
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        Y = np.eye(len(self.classes_))[y_enc]
        self.W, self.b = np.zeros((X.shape[1], Y.shape[1])), np.zeros(Y.shape[1])
        for _ in range(self.n_iters):
            Z = X @ self.W + self.b
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            self.W -= self.lr * X.T @ (P - Y) / len(X)
            self.b -= self.lr * (P - Y).mean(axis=0)
        return self

    def predict(self, X):
        return self.classes_[(X @ self.W + self.b).argmax(axis=1)]


class StackingClassifier:
    """Instead of a fixed vote, LEARN how to combine the base models.

    1. Split the data into `cv` folds. For each fold, train every base model on the other folds
       and predict this fold. These out-of-fold predictions become the meta-features.
    2. Train a final (meta) model on [meta-features -> true labels].
    3. Refit the base models on ALL the data; at prediction time their outputs feed the meta model.

    Out-of-fold predictions are essential: if the meta model were trained on predictions the base
    models made for rows they were trained on, it would learn to over-trust whichever base model
    memorizes best.

    Assumes every training fold contains every class (use plenty of data per class, small cv)."""

    def __init__(self, estimators, final_estimator=None, cv=5, random_state=0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, k = len(X), len(self.classes_)

        folds = np.array_split(np.random.default_rng(self.random_state).permutation(n), self.cv)
        meta = np.zeros((n, k * len(self.estimators)))
        for held_out in folds:
            train = np.setdiff1d(np.arange(n), held_out)
            for j, (_, est) in enumerate(self.estimators):
                model = copy.deepcopy(est).fit(X[train], y[train])
                meta[held_out, j * k:(j + 1) * k] = _probabilities(model, X[held_out], self.classes_)

        self.models_ = [copy.deepcopy(est).fit(X, y) for _, est in self.estimators]
        self.final_estimator_ = copy.deepcopy(self.final_estimator or _Softmax()).fit(meta, y)
        return self

    def _meta_features(self, X):
        return np.hstack([_probabilities(m, X, self.classes_) for m in self.models_])

    def predict(self, X):
        return self.final_estimator_.predict(self._meta_features(np.asarray(X, dtype=float)))
