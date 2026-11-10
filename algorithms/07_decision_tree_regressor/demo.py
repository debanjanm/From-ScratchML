import numpy as np
from model import DecisionTreeRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SklearnTree

X, y = make_friedman1(n_samples=500, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ours = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
theirs = SklearnTree(max_depth=5, random_state=0).fit(X_train, y_train)

mse_ours = np.mean((ours.predict(X_test) - y_test) ** 2)
mse_theirs = np.mean((theirs.predict(X_test) - y_test) ** 2)
print("friedman1 test MSE: ours=%.4f sklearn=%.4f" % (mse_ours, mse_theirs))
assert abs(mse_ours - mse_theirs) / mse_theirs < 0.05
