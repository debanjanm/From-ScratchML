from model import KernelSVM
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

C, gamma = 1.0, 1.0
ours = KernelSVM(kernel="rbf", C=C, gamma=gamma).fit(X_train, y_train)
theirs = SVC(kernel="rbf", C=C, gamma=gamma).fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("moons test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
print("support vectors: ours=%d sklearn=%d (of %d training points)"
      % (len(ours.support_vectors_), len(theirs.support_vectors_), len(X_train)))
assert abs(acc_ours - acc_theirs) < 0.03

linear = KernelSVM(kernel="linear", C=C).fit(X_train, y_train)
print("same data, linear kernel:   %.4f" % (linear.predict(X_test) == y_test).mean())
