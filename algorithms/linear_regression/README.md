# Linear Regression

Fits `y = Xw + b` by minimizing mean squared error.

## Methods

- **`normal_equation`** (default): closed-form solution `w = (XᵀX)⁻¹Xᵀy` (via pseudo-inverse for numerical stability).
- **`gradient_descent`**: iterative batch gradient descent on the MSE loss, useful when `X` is large or `XᵀX` is ill-conditioned.

## Usage

```python
from model import LinearRegression

model = LinearRegression(method="normal_equation").fit(X, y)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
