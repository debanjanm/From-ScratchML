from model import BaggingClassifier, _Tree
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier as SklearnBagging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

ours = BaggingClassifier(n_estimators=50).fit(X_train, y_train)
theirs = SklearnBagging(DecisionTreeClassifier(), n_estimators=50, oob_score=True, random_state=0).fit(X_train, y_train)
single = _Tree(n_classes=2).fit(X_train, y_train)

acc = lambda pred: (pred == y_test).mean()
print("breast cancer test accuracy: ours=%.4f sklearn=%.4f  (one tree alone: %.4f)"
      % (acc(ours.predict(X_test)), acc(theirs.predict(X_test)), acc(single.predict(X_test))))
print("out-of-bag score:            ours=%.4f sklearn=%.4f" % (ours.oob_score_, theirs.oob_score_))
assert abs(acc(ours.predict(X_test)) - acc(theirs.predict(X_test))) < 0.04
assert abs(ours.oob_score_ - theirs.oob_score_) < 0.03
