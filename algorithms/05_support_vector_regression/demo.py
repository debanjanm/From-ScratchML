import numpy as np
from model import LinearSVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X, y = make_regression(n_samples=400, n_features=5, n_informative=3, noise=10.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

x_scaler = StandardScaler().fit(X_train)
X_train, X_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
y_std = y_train.std()
y_train, y_test = y_train / y_std, y_test / y_std  # epsilon is in y's units: put y on unit scale

C, epsilon = 10.0, 0.1
ours = LinearSVR(epsilon=epsilon, lam=1 / (C * len(y_train)), n_iters=3000).fit(X_train, y_train)
theirs = SVR(kernel="linear", C=C, epsilon=epsilon).fit(X_train, y_train)  # lam = 1 / (C * n)

mse_ours = np.mean((ours.predict(X_test) - y_test) ** 2)
mse_theirs = np.mean((theirs.predict(X_test) - y_test) ** 2)
print("test MSE (y scaled): ours=%.4f sklearn=%.4f" % (mse_ours, mse_theirs))
print("weights ours   :", np.round(ours.weights, 3))
print("weights sklearn:", np.round(theirs.coef_[0], 3))
assert abs(mse_ours - mse_theirs) < 0.01
