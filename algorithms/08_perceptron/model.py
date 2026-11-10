import numpy as np


class Perceptron:
    """The original linear classifier (Rosenblatt, 1958). For every misclassified sample, nudge the
    boundary toward it:  w += lr * y * x.  Converges only if the classes are linearly separable.

    Binary; labels can be any two values."""

    def __init__(self, lr=1.0, epochs=100, random_state=0):
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.classes_ = None
        self.weights = None
        self.bias = 0.0
        self.errors_ = []  # misclassifications per epoch; ends at 0 once converged

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Perceptron is binary: needs exactly 2 classes")
        y_pm = 2 * y_enc - 1  # {0, 1} -> {-1, +1}

        rng = np.random.default_rng(self.random_state)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for i in rng.permutation(len(X)):
                if y_pm[i] * (X[i] @ self.weights + self.bias) <= 0:  # wrong side (or on the line)
                    self.weights += self.lr * y_pm[i] * X[i]
                    self.bias += self.lr * y_pm[i]
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
