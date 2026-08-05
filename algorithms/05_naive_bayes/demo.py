from model import GaussianNaiveBayes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SklearnGNB

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

ours = GaussianNaiveBayes().fit(X_train, y_train)
theirs = SklearnGNB().fit(X_train, y_train)

acc_ours = (ours.predict(X_test) == y_test).mean()
acc_theirs = (theirs.predict(X_test) == y_test).mean()
print("iris test accuracy: ours=%.4f sklearn=%.4f" % (acc_ours, acc_theirs))
assert acc_ours == acc_theirs, "Gaussian NB should match sklearn exactly on iris"
