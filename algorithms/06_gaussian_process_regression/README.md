# Gaussian Process Regression

A non-parametric, Bayesian approach: instead of learning weights, put a prior over *functions* and condition it on the data. The result is a prediction **and an uncertainty** at every point.

With an RBF kernel `k(a, b) = variance · exp(−‖a − b‖² / 2·length_scale²)` (nearby inputs → similar outputs) and `K = k(X, X) + noise²·I`:

```
mean(x*) = k(x*, X) · K⁻¹ y
var(x*)  = k(x*, x*) − k(x*, X) · K⁻¹ · k(X, x*)
```

K is factored with a Cholesky decomposition rather than inverted directly. Near training points the std shrinks; far from them the prediction reverts to the prior (mean 0, std `√variance`).

- `length_scale`: how quickly the function wiggles. `variance`: output scale. `noise`: observation noise std.
- Hyperparameters are fixed here; picking them by maximizing marginal likelihood is a natural extension.
- Cost is O(n³), so it suits small datasets.

## Usage

```python
from model import GaussianProcessRegressor

gp = GaussianProcessRegressor(length_scale=1.0, variance=1.0, noise=0.1).fit(X_train, y_train)
mean, std = gp.predict(X_test, return_std=True)
```

## Run

```bash
python test_model.py
python demo.py
```
