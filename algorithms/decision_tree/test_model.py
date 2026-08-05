import numpy as np
from model import DecisionTreeClassifier, _gini


def test_gini_pure_and_mixed():
    assert _gini(np.array([0, 0, 0])) == 0.0
    assert _gini(np.array([0, 1])) == 0.5


def test_simple_threshold_split():
    X = np.array([[1], [2], [3], [8], [9], [10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = model.predict(X)
    assert (preds == y).all()


def test_iris_like_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0
    )
    model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    assert accuracy > 0.9, accuracy


if __name__ == "__main__":
    test_gini_pure_and_mixed()
    test_simple_threshold_split()
    test_iris_like_accuracy()
    print("all tests passed")
