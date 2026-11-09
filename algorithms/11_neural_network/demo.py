from model import NeuralNetClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = load_digits()
X = data.data / 16.0  # pixel values 0..16 -> 0..1
X_train, X_test, y_train, y_test = train_test_split(X, data.target, test_size=0.3, random_state=42)

ours = NeuralNetClassifier(hidden_sizes=(32,), lr=0.1, epochs=50, batch_size=32).fit(X_train, y_train)
theirs = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("digits test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
print("loss: %.4f -> %.4f over %d epochs" % (ours.loss_history_[0], ours.loss_history_[-1], len(ours.loss_history_)))
assert abs(acc_ours - acc_theirs) < 0.04
