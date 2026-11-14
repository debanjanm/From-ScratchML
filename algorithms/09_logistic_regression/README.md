# Logistic Regression

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Binary classifier: `p(y=1|x) = sigmoid(Xw + b)`, trained by batch gradient descent minimizing binary cross-entropy.

## Usage

```python
# (the class lives in notebook.ipynb)

model = LogisticRegression(lr=0.1, n_iters=2000).fit(X, y)
probs = model.predict_proba(X_test)
labels = model.predict(X_test)
```
