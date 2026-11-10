import numpy as np
from model import NeuralNetClassifier


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    Y = np.eye(4)[rng.integers(0, 4, 8)]
    net = NeuralNetClassifier(hidden_sizes=(5, 4))
    net._init_params(3, 4, rng)
    # zero biases can put a pre-activation exactly on the ReLU kink, where finite differences lie
    net.b = [rng.normal(size=b.shape) for b in net.b]

    _, grad_W, grad_b = net._loss_and_grads(X, Y)
    eps = 1e-6
    for params, grads in ((net.W, grad_W), (net.b, grad_b)):
        for layer, grad in zip(params, grads):
            for idx in np.ndindex(layer.shape):
                original = layer[idx]
                layer[idx] = original + eps
                up = net._loss_and_grads(X, Y)[0]
                layer[idx] = original - eps
                down = net._loss_and_grads(X, Y)[0]
                layer[idx] = original
                assert abs((up - down) / (2 * eps) - grad[idx]) < 1e-6, idx


def test_learns_xor():
    # not linearly separable: needs the hidden layer
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    y = np.array([0, 1, 1, 0] * 25)
    net = NeuralNetClassifier(hidden_sizes=(8,), lr=0.1, epochs=300, batch_size=16).fit(X, y)
    assert (net.predict(X) == y).all()


def test_iris_accuracy_and_loss_decreases():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    net = NeuralNetClassifier(hidden_sizes=(16,), epochs=100).fit(scaler.transform(X_train), y_train)
    assert net.loss_history_[-1] < net.loss_history_[0]
    assert (net.predict(scaler.transform(X_test)) == y_test).mean() > 0.9


if __name__ == "__main__":
    test_backprop_matches_numerical_gradient()
    test_learns_xor()
    test_iris_accuracy_and_loss_decreases()
    print("all tests passed")
