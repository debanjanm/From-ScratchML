from model import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

ours = AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
theirs = SklearnAdaBoost(
    estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, algorithm="SAMME", random_state=0
).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("breast cancer test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
assert abs(acc_ours - acc_theirs) < 0.03
