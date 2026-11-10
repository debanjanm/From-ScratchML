import numpy as np
from model import BernoulliNB, MultinomialNB

# tiny "documents": columns = [free, money, meeting, report]
WORD_COUNTS = np.array(
    [[3, 2, 0, 0], [2, 3, 0, 1], [4, 1, 0, 0],   # spam: free / money
     [0, 0, 3, 2], [0, 1, 2, 3], [0, 0, 4, 1]],  # ham: meeting / report
    dtype=float,
)
LABELS = np.array(["spam", "spam", "spam", "ham", "ham", "ham"])


def test_multinomial_classifies_word_counts():
    model = MultinomialNB().fit(WORD_COUNTS, LABELS)
    assert (model.predict(WORD_COUNTS) == LABELS).all()
    assert model.predict(np.array([[5.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 3.0]])).tolist() == ["spam", "ham"]


def test_bernoulli_classifies_presence_absence():
    X = (WORD_COUNTS > 0).astype(float)
    model = BernoulliNB().fit(X, LABELS)
    assert (model.predict(X) == LABELS).all()


def test_smoothing_handles_an_unseen_word():
    # "report" never appears in spam training docs: without smoothing P = 0 and log(0) = -inf
    X = np.array([[2.0, 1.0, 0.0], [3.0, 2.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 2.0]])
    y = np.array([0, 0, 1, 1])
    model = MultinomialNB(alpha=1.0).fit(X, y)
    assert np.isfinite(model.log_prob).all()
    assert model.predict(np.array([[1.0, 1.0, 1.0]])).shape == (1,)


if __name__ == "__main__":
    test_multinomial_classifies_word_counts()
    test_bernoulli_classifies_presence_absence()
    test_smoothing_handles_an_unseen_word()
    print("all tests passed")
