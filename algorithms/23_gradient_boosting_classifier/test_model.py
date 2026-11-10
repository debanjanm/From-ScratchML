import numpy as np
from model import GradientBoostingClassifier


def test_zero_rounds_predicts_the_class_priors():
    X = np.zeros((10, 2))
    y = np.array([0] * 7 + [1] * 3)
    probs = GradientBoostingClassifier(n_estimators=0).fit(X, y).predict_proba(X)
    assert np.allclose(probs, [0.7, 0.3])


def test_binary_xor_like_problem_needs_interactions():
    # XOR: boosted depth-2 trees can model it, a linear model cannot
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=60, max_depth=2).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.97


def test_multiclass_string_labels_probabilities_and_loss():
    rng = np.random.default_rng(1)
    centers = np.array([[-4, 0], [4, 0], [0, 5]])
    X = np.vstack([rng.normal(c, 0.8, (50, 2)) for c in centers])
    y = np.repeat(np.array(["red", "green", "blue"]), 50)
    model = GradientBoostingClassifier(n_estimators=40).fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (150, 3) and np.allclose(probs.sum(axis=1), 1.0)
    assert (model.predict(X) == y).mean() > 0.98
    assert model.loss_history_[-1] < 0.2 * model.loss_history_[0]  # training deviance falls fast


if __name__ == "__main__":
    test_zero_rounds_predicts_the_class_priors()
    test_binary_xor_like_problem_needs_interactions()
    test_multiclass_string_labels_probabilities_and_loss()
    print("all tests passed")
