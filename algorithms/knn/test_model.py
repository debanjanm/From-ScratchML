import numpy as np
from model import KNNClassifier


def test_simple_separated_clusters():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    model = KNNClassifier(k=3).fit(X_train, y_train)

    preds = model.predict(np.array([[0.5, 0.5], [10.5, 10.5]]))
    assert list(preds) == [0, 1]


def test_iris_like_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0
    )
    model = KNNClassifier(k=5).fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    assert accuracy > 0.9, accuracy


if __name__ == "__main__":
    test_simple_separated_clusters()
    test_iris_like_accuracy()
    print("all tests passed")
