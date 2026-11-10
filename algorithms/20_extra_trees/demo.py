from model import ExtraTreesClassifier
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTrees
from sklearn.model_selection import train_test_split

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

ours = ExtraTreesClassifier(n_estimators=100).fit(X_train, y_train)
theirs = SklearnExtraTrees(n_estimators=100, random_state=0).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("wine test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
assert abs(acc_ours - acc_theirs) < 0.06  # random forests of any kind: close, not identical
