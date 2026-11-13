# Bayesian Linear Regression

Ordinary linear regression (`01_linear_regression`) returns **one** weight vector and predictions with no sense of how sure it is. Bayesian linear regression keeps a whole **distribution** over the weights, so every prediction comes with an error bar that is honest about both noise in the data and ignorance about the model.

## How it works

Assume Gaussians on both sides:

- noise: `y = Xw + ε`, `ε ~ N(0, 1/α)` (`α` = noise precision)
- prior: `w ~ N(0, (1/λ)·I)` (`λ` = weight precision)

Then the **posterior** over `w` is Gaussian in closed form:

```
S = (λI + α XᵀX)⁻¹        posterior covariance
m = α S Xᵀy                posterior mean   (= ridge regression with strength λ/α)
```

and the **prediction** at `x` is Gaussian too:

```
mean = x·m        variance = 1/α  +  xᵀ S x
                            └ noise   └ uncertainty about the weights — grows away from the training data
```

### No hand-tuned regularization

`α` and `λ` are learned by maximising the **marginal likelihood** (evidence) with MacKay's fixed-point updates:

```
γ = Σ αeᵢ / (λ + αeᵢ)        (effective number of well-determined weights; eᵢ = eigenvalues of XᵀX)
λ ← γ / mᵀm
α ← (N − γ) / ‖y − Xm‖²
```

Repeat until stable. Too many features and too little data? The evidence shrinks the useless weights automatically — no cross-validation. Names `alpha_`, `lambda_` match scikit-learn's `BayesianRidge`, and with its priors set to zero the results are identical (the demo checks it).

Linear in the *weights* only: feed it polynomial or other basis features to fit curves (demo does a cubic and prints widening intervals at `x = 0.9, 1.5, 3`).

`ponytail:` one global weight precision `λ`; per-feature precisions (Automatic Relevance Determination) is the upgrade for sparse problems.

## Usage

```python
from model import BayesianLinearRegression

model = BayesianLinearRegression().fit(X, y)
mean, std = model.predict(X_new, return_std=True)   # std includes the noise
model.coef_, model.sigma_                           # posterior mean and covariance of the weights
model.alpha_, model.lambda_                         # learned precisions
```

## Run

```bash
python test_model.py
python demo.py
```
