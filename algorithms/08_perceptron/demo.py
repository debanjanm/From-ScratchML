from model import Perceptron
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=300, centers=[[-3, -3], [3, 3]], cluster_std=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ours = Perceptron(epochs=100).fit(X_train, y_train)
theirs = SklearnPerceptron(max_iter=1000, tol=None, random_state=0).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
print("mistakes per epoch:", ours.errors_)  # reaches 0 = converged (data is linearly separable)
assert ours.errors_[-1] == 0
assert abs(acc_ours - acc_theirs) < 0.05
