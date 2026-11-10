from model import SoftmaxRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

ours = SoftmaxRegression(lr=0.1, n_iters=2000).fit(X_train, y_train)
theirs = LogisticRegression(penalty=None, max_iter=2000).fit(X_train, y_train)  # multinomial by default

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("iris (3 classes) test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
print("loss: %.4f -> %.4f" % (ours.loss_history_[0], ours.loss_history_[-1]))
assert abs(acc_ours - acc_theirs) < 0.03
