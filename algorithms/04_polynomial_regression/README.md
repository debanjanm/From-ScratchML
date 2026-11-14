# Polynomial Regression

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Fits curves with a linear model by expanding the inputs: for one feature `x` and degree 3 the model sees `[x, x², x³]` and learns a weight for each. It is still **linear in the weights**, so the least-squares solution from `01_linear_regression` applies unchanged — only the features change.

With several features, cross terms are included (`a·b`, `a²·b`, ...), so the feature count grows fast with degree.

Higher degree fits training data better but overfits; pair with Ridge (`02_ridge_lasso`) for high degrees. Scale features first for degree > 3 or wide-range data.

## Usage

```python
# (the class lives in notebook.ipynb)

model = PolynomialRegression(degree=3).fit(X_train, y_train)
preds = model.predict(X_test)
```
