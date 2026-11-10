from model import StackingClassifier, VotingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import StackingClassifier as SklearnStacking
from sklearn.ensemble import VotingClassifier as SklearnVoting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# the base learners are plugged in from outside: here three different scikit-learn models
base = [
    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(7))),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
]
acc = lambda model: (model.predict(X_test) == y_test).mean()

for name, model in base:
    print("alone  %-7s %.4f" % (name, acc(model.fit(X_train, y_train))))

for voting in ("hard", "soft"):
    ours = VotingClassifier(base, voting=voting).fit(X_train, y_train)
    theirs = SklearnVoting(base, voting=voting).fit(X_train, y_train)
    print("%s voting:    ours=%.4f sklearn=%.4f" % (voting, acc(ours), acc(theirs)))
    assert abs(acc(ours) - acc(theirs)) < 0.01

ours = StackingClassifier(base, cv=5).fit(X_train, y_train)
theirs = SklearnStacking(base, cv=5).fit(X_train, y_train)
print("stacking:     ours=%.4f sklearn=%.4f" % (acc(ours), acc(theirs)))
assert abs(acc(ours) - acc(theirs)) < 0.03
