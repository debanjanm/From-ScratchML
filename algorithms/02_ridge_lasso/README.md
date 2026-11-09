# Ridge and Lasso Regression

Linear regression with a penalty on the weights, to fight overfitting. The bias is never penalized (data is centered first). Objectives follow scikit-learn's conventions so coefficients are directly comparable.

| | Penalty | Objective | Solver | Effect |
|---|---|---|---|---|
| **Ridge** | L2 | `‖y − Xw‖² + α‖w‖²` | closed form `(XᵀX + αI)⁻¹Xᵀy` | shrinks all weights smoothly |
| **Lasso** | L1 | `1/(2n)‖y − Xw‖² + α‖w‖₁` | coordinate descent + soft-thresholding | sets some weights exactly to 0 (feature selection) |

`alpha=0` for Ridge is ordinary least squares (`01_linear_regression`). Larger `alpha` means stronger shrinkage.

Standardize features first if they have different scales — the penalty treats all weights equally.

## Usage

```python
from model import RidgeRegression, LassoRegression

ridge = RidgeRegression(alpha=10.0).fit(X_train, y_train)
lasso = LassoRegression(alpha=1.0).fit(X_train, y_train)
print(lasso.weights)          # irrelevant features come out as exactly 0
preds = ridge.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
