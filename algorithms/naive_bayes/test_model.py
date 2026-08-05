import numpy as np
from model import GaussianNaiveBayes


def test_separated_clusters():
    rng = np.random.default_rng(0)
    class0 = rng.normal(loc=-5, scale=0.5, size=(100, 2))
    class1 = rng.normal(loc=5, scale=0.5, size=(100, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 100 + [1] * 100)

    model = GaussianNaiveBayes().fit(X, y)
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.98, accuracy


def test_iris_like_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0
    )
    model = GaussianNaiveBayes().fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    assert accuracy > 0.9, accuracy


if __name__ == "__main__":
    test_separated_clusters()
    test_iris_like_accuracy()
    print("all tests passed")
