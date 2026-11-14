# Decision Tree Regressor

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Same recursive splitting as the classifier (`15_decision_tree`), but for a continuous target:

- a split is chosen to minimize the squared error of the two sides (variance reduction)
- a leaf predicts the **mean** of its training targets, so predictions are piecewise constant

Splits are found in one pass per feature using cumulative sums over the sorted values, so a fit is O(n log n) per feature rather than re-scanning for every threshold. This tree is also the building block of `22_gradient_boosting`.

`max_depth` controls the bias/variance trade-off: shallow trees underfit, deep trees memorize noise.

## Usage

```python
# (the class lives in notebook.ipynb)

model = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
preds = model.predict(X_test)
```
