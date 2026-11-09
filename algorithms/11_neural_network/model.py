import numpy as np


class NeuralNetClassifier:
    """Feedforward network: ReLU hidden layers, softmax output, cross-entropy loss,
    backpropagation written by hand, trained with mini-batch gradient descent."""

    def __init__(self, hidden_sizes=(16,), lr=0.1, epochs=100, batch_size=32, random_state=0):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.classes_ = None
        self.W, self.b = [], []
        self.loss_history_ = []

    def _init_params(self, n_in, n_out, rng):
        sizes = [n_in, *self.hidden_sizes, n_out]
        # He initialization keeps ReLU activations from shrinking/exploding with depth
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X):
        acts, pre = [X], []  # acts[i] is the input of layer i; pre[i] its pre-activation
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            pre.append(z)
            a = z if i == len(self.W) - 1 else np.maximum(z, 0)  # last layer stays as logits
            acts.append(a)
        e = np.exp(a - a.max(axis=1, keepdims=True))  # subtract max: numerically stable softmax
        return acts, pre, e / e.sum(axis=1, keepdims=True)

    def _loss_and_grads(self, X, Y):
        acts, pre, probs = self._forward(X)
        n = len(X)
        loss = -np.sum(Y * np.log(probs + 1e-12)) / n

        grad_W, grad_b = [None] * len(self.W), [None] * len(self.W)
        dz = (probs - Y) / n  # d(loss)/d(logits) for softmax + cross-entropy
        for i in reversed(range(len(self.W))):
            grad_W[i] = acts[i].T @ dz
            grad_b[i] = dz.sum(axis=0)
            if i > 0:
                dz = (dz @ self.W[i].T) * (pre[i - 1] > 0)  # through weights, then ReLU gate
        return loss, grad_W, grad_b

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        Y = np.eye(len(self.classes_))[y_enc]  # one-hot targets

        rng = np.random.default_rng(self.random_state)
        self._init_params(X.shape[1], len(self.classes_), rng)
        self.loss_history_ = []

        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = order[start : start + self.batch_size]
                _, grad_W, grad_b = self._loss_and_grads(X[idx], Y[idx])
                for i in range(len(self.W)):
                    self.W[i] -= self.lr * grad_W[i]
                    self.b[i] -= self.lr * grad_b[i]
            self.loss_history_.append(self._loss_and_grads(X, Y)[0])
        return self

    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float))[2]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
