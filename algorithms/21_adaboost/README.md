# AdaBoost

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Boosting with decision **stumps** (one-split trees), each barely better than guessing. Rounds are sequential:

1. Train a stump on the weighted data; compute its weighted error `err`.
2. Give it a vote strength `alpha = ½ ln((1 − err) / err)` — more accurate stumps count more.
3. Re-weight samples: `w *= exp(−alpha · y · pred)`. Misclassified samples grow, so the next stump focuses on them.

Prediction is `sign(Σ alpha · stump(x))`.

Compare `22_gradient_boosting` (fits residuals) and `19_random_forest` (independent trees, parallel). Binary classification only; labels can be any two values.

## Usage

```python
# (the class lives in notebook.ipynb)

model = AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
preds = model.predict(X_test)
scores = model.decision_function(X_test)   # signed vote total
```
