from model import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.model_selection import train_test_split

data = load_wine()  # 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

ours = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
theirs = SklearnGBC(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("wine (3 classes) test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
print("training log-loss: %.4f -> %.4f over %d rounds"
      % (ours.loss_history_[0], ours.loss_history_[-1], len(ours.loss_history_)))
assert abs(acc_ours - acc_theirs) < 0.04
