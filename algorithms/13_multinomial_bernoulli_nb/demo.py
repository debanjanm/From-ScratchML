from model import BernoulliNB, MultinomialNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

# digit images: pixel intensities 0..16 behave like counts; thresholded they become binary features
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

ours = MultinomialNB().fit(X_train, y_train)
theirs = SklearnMultinomialNB().fit(X_train, y_train)
print("Multinomial  test accuracy: ours=%.4f sklearn=%.4f"
      % ((ours.predict(X_test) == y_test).mean(), (theirs.predict(X_test) == y_test).mean()))
assert (ours.predict(X_test) == theirs.predict(X_test)).all()

Xb_train, Xb_test = (X_train > 8).astype(float), (X_test > 8).astype(float)
ours = BernoulliNB().fit(Xb_train, y_train)
theirs = SklearnBernoulliNB(binarize=None).fit(Xb_train, y_train)  # already binary
print("Bernoulli    test accuracy: ours=%.4f sklearn=%.4f"
      % ((ours.predict(Xb_test) == y_test).mean(), (theirs.predict(Xb_test) == y_test).mean()))
assert (ours.predict(Xb_test) == theirs.predict(Xb_test)).all()
