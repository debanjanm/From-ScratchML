# XGBoost (from scratch)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Gradient boosting (`22_gradient_boosting`, `23_gradient_boosting_classifier`) taken to second order, plus regularization. Each sample has a **gradient** `g` and a **hessian** `h` of the loss at the current prediction:

| objective | `g` | `h` |
|---|---|---|
| squared error | `pred − y` | `1` |
| logistic | `p − y` | `p(1 − p)` |

For a leaf containing a set of samples with `G = Σg`, `H = Σh`, the loss-minimizing weight and its quality score are:

```
w = −G / (H + λ)            score = G² / (H + λ)
```

A split is accepted only if it improves the total score by more than the price `γ` of an extra leaf:

```
gain = ½ · [ score_left + score_right − score_parent ] − γ
```

What this adds over plain gradient boosting:

- **`reg_lambda` (λ):** L2 penalty on leaf weights → smaller, safer updates.
- **`gamma` (γ):** minimum gain to split → built-in pruning; a large γ gives a constant model.
- **`min_child_weight`:** minimum Σh in a child → stops splits on tiny, noisy groups.
- **Newton step:** using `h` makes each tree a better step than the plain gradient.

Missing from the real library (listed in the code): row/column subsampling, histogram splits, missing-value handling, multiclass, GPU/parallelism. The demo compares against the real `xgboost` when it is installed (optional), otherwise against scikit-learn.

## Usage

```python
# (the class lives in notebook.ipynb)

reg = XGBoost("reg:squarederror", n_estimators=100, learning_rate=0.3, max_depth=3).fit(X_train, y_train)
clf = XGBoost("binary:logistic", reg_lambda=1.0, gamma=0.1).fit(X_train, y_train)
preds, probs = clf.predict(X_test), clf.predict_proba(X_test)
```
