# Linear Regression

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Fits `y = Xw + b` by minimizing mean squared error.

## Methods

- **`normal_equation`** (default): closed-form solution `w = (XᵀX)⁻¹Xᵀy` (via pseudo-inverse for numerical stability).
- **`gradient_descent`**: iterative batch gradient descent on the MSE loss, useful when `X` is large or `XᵀX` is ill-conditioned.

## Usage

```python
# (the class lives in notebook.ipynb)

model = LinearRegression(method="normal_equation").fit(X, y)
preds = model.predict(X_test)
```
