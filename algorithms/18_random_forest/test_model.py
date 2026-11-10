import numpy as np
from model import RandomForestClassifier


def test_separated_clusters_with_string_labels():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.5, (60, 3)), rng.normal(3, 0.5, (60, 3))])
    y = np.array(["cat"] * 60 + ["dog"] * 60)

    model = RandomForestClassifier(n_estimators=10).fit(X, y)
    preds = model.predict(X)
    assert set(preds) <= {"cat", "dog"}
    assert (preds == y).mean() > 0.98


def test_iris_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0
    )
    model = RandomForestClassifier(n_estimators=25, random_state=0).fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() > 0.9


def test_same_seed_same_forest():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    a = RandomForestClassifier(n_estimators=5, random_state=7).fit(X, y).predict(X)
    b = RandomForestClassifier(n_estimators=5, random_state=7).fit(X, y).predict(X)
    assert (a == b).all()


if __name__ == "__main__":
    test_separated_clusters_with_string_labels()
    test_iris_accuracy()
    test_same_seed_same_forest()
    print("all tests passed")
