from model import LinearSVM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)
scaler = StandardScaler().fit(X_train)  # SVMs need scaled features
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

ours = LinearSVM(lam=0.01, n_iters=1000).fit(X_train, y_train)
theirs = SVC(kernel="linear", C=1.0).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("breast cancer test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
assert abs(acc_ours - acc_theirs) < 0.03
