# Kernel SVM (SMO)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

The linear SVM (`16_svm`) can only draw straight boundaries. The **kernel trick** fixes that: replace every dot product `x·x'` with a kernel `k(x, x')` that equals a dot product in some much richer feature space — without ever constructing that space.

| Kernel | `k(x, x')` | Boundary |
|---|---|---|
| `linear` | `x·x'` | straight line |
| `poly` | `(γ·x·x' + 1)^degree` | polynomial curve |
| `rbf` | `exp(−γ‖x − x'‖²)` | flexible, local — larger `γ` means wigglier |

## Dual form and SMO

Instead of learning `w`, the dual learns one multiplier `alpha_i` per training point (`0 ≤ alpha ≤ C`, `Σ alpha·y = 0`). Points with `alpha > 0` are the **support vectors**, and prediction uses only them:

```
f(x) = Σ alpha_i · y_i · k(x_i, x) + b
```

**SMO** solves the dual by repeatedly choosing two multipliers and optimizing just those two analytically (the equality constraint forces them to move together), until no point violates the KKT conditions by more than `tol`. This is a simplified version: the second multiplier is chosen at random rather than by a heuristic.

`C` is the penalty for margin violations (large = harder margin, risk of overfitting). Scale features first; `gamma` is on the scale of `1 / ‖x − x'‖²`.

## Usage

```python
# (the class lives in notebook.ipynb)

model = KernelSVM(kernel="rbf", C=1.0, gamma=1.0).fit(X_train, y_train)
preds = model.predict(X_test)
print(len(model.support_vectors_))
```
