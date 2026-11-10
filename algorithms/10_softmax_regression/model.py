import numpy as np


def _softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))  # subtract the row max: numerically stable
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Multiclass logistic regression: P(class k | x) = softmax(xW + b)_k,
    trained by batch gradient descent on cross-entropy loss. Any number of classes, any label type."""

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.classes_ = None
        self.W = None
        self.b = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        Y = np.eye(k)[y_enc]  # one-hot targets

        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        self.loss_history_ = []
        for _ in range(self.n_iters):
            P = _softmax(X @ self.W + self.b)
            self.loss_history_.append(-np.sum(Y * np.log(P + 1e-12)) / n)
            error = P - Y  # gradient of cross-entropy w.r.t. the logits
            self.W -= self.lr * (X.T @ error) / n
            self.b -= self.lr * error.mean(axis=0)
        return self

    def predict_proba(self, X):
        return _softmax(np.asarray(X, dtype=float) @ self.W + self.b)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
