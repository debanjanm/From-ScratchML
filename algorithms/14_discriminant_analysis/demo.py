from model import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA
from sklearn.model_selection import train_test_split

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

for name, ours, theirs in (
    ("LDA", LinearDiscriminantAnalysis(), SklearnLDA()),
    ("QDA", QuadraticDiscriminantAnalysis(), SklearnQDA()),
):
    ours.fit(X_train, y_train)
    theirs.fit(X_train, y_train)
    acc_ours = (ours.predict(X_test) == y_test).mean()
    acc_theirs = (theirs.predict(X_test) == y_test).mean()
    print("wine %s test accuracy: ours=%.4f sklearn=%.4f" % (name, acc_ours, acc_theirs))
    assert (ours.predict(X_test) == theirs.predict(X_test)).all()
