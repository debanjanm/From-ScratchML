# Multinomial and Bernoulli Naive Bayes

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Same idea as `12_naive_bayes` (Bayes' rule + assume features are independent given the class), but for **discrete** features instead of continuous Gaussian ones — the standard choice for text.

| | Features | Likelihood of a sample | Absence is evidence? |
|---|---|---|---|
| **MultinomialNB** | counts (word counts, TF) | `Π p_j^(x_j)` | no — only what occurs counts |
| **BernoulliNB** | binary (word present / absent) | `Π p_j^(x_j) · (1 − p_j)^(1 − x_j)` | yes |

Both use **Laplace smoothing** (`alpha=1`): add `alpha` to every count so a word never seen in a class gets a small probability instead of 0, which would otherwise veto the whole class (`log 0 = −∞`).

Everything is done in log space (sums of logs, not products of tiny probabilities) to avoid underflow.

Rule of thumb: use Multinomial when word *frequency* matters (long documents), Bernoulli for short texts or when absence is informative.

## Usage

```python
# (the class lives in notebook.ipynb)

counts_model = MultinomialNB(alpha=1.0).fit(X_counts, y)
binary_model = BernoulliNB(alpha=1.0).fit((X_counts > 0).astype(float), y)
preds = counts_model.predict(X_new)
```
