# Elastic Net

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Ridge and Lasso combined. The penalty is a blend controlled by `l1_ratio`:

```
1/(2n)‖y − Xw‖²  +  α·l1_ratio·‖w‖₁  +  ½·α·(1 − l1_ratio)·‖w‖²
```

- `l1_ratio=1` → Lasso (`02_ridge_lasso`), `l1_ratio=0` → Ridge.
- The L1 part still zeroes out weak features.
- The L2 part keeps **correlated features together** instead of arbitrarily picking one — Lasso's main weakness.

Same coordinate-descent solver as Lasso; the only change is the L2 term added to the update's denominator. Standardize features first if scales differ.

## Usage

```python
# (the class lives in notebook.ipynb)

model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train)
preds = model.predict(X_test)
```
