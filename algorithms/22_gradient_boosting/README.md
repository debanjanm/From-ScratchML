# Gradient Boosting (regression)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Builds an additive model `F(x) = mean(y) + lr * sum(tree_m(x))`. Each shallow regression tree is fit to the residuals `y - F(x)` of the ensemble so far, which is the negative gradient of squared error. Small `learning_rate` with many trees generalizes best.

Unlike Random Forest (parallel, variance-reducing), trees here are sequential and bias-reducing.

## Usage

```python
# (the class lives in notebook.ipynb)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
preds = model.predict(X_test)
```
