import numpy as np
from model import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
class0 = rng.normal(loc=-2, scale=1.2, size=(200, 2))
class1 = rng.normal(loc=2, scale=1.2, size=(200, 2))
X = np.vstack([class0, class1])
y = np.array([0] * 200 + [1] * 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

ours = LogisticRegression(lr=0.1, n_iters=3000).fit(X_train, y_train)
theirs = SklearnLogisticRegression().fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
assert abs(acc_ours - acc_theirs) < 0.05

try:
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15, alpha=0.6)
    plt.title("Logistic Regression data (from scratch model trained on this)")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
