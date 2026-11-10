import numpy as np
from model import ExtraTreesClassifier, _ExtraTree


def test_string_labels_and_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.6, (60, 3)), rng.normal(3, 0.6, (60, 3))])
    y = np.array(["cat"] * 60 + ["dog"] * 60)
    model = ExtraTreesClassifier(n_estimators=20).fit(X, y)
    preds = model.predict(X)
    assert set(preds) <= {"cat", "dog"}
    assert (preds == y).mean() > 0.98


def test_thresholds_are_random_not_optimized():
    # one clean split exists at ~5; different seeds must still pick different cut points
    X = np.array([[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    cuts = set()
    for seed in range(8):
        tree = _ExtraTree(1, 2, 1, 2, np.random.default_rng(seed)).fit(X, y)
        cuts.add(round(tree.root[1], 3))
    assert len(cuts) > 4
    assert all(1.0 <= c <= 9.0 for c in cuts)


def test_iris_accuracy_and_determinism():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
    a = ExtraTreesClassifier(n_estimators=40, random_state=3).fit(X_train, y_train)
    b = ExtraTreesClassifier(n_estimators=40, random_state=3).fit(X_train, y_train)
    assert (a.predict(X_test) == y_test).mean() > 0.9
    assert (a.predict(X_test) == b.predict(X_test)).all()


if __name__ == "__main__":
    test_string_labels_and_separable_data()
    test_thresholds_are_random_not_optimized()
    test_iris_accuracy_and_determinism()
    print("all tests passed")
