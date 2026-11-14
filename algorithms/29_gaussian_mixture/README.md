# Gaussian Mixture Model (EM)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Model the data as coming from `k` Gaussians, each with its own **weight, mean and full covariance**. This is K-Means (`27_kmeans`) made probabilistic:

| | K-Means | Gaussian Mixture |
|---|---|---|
| Assignment | hard (one cluster) | **soft** — a probability per cluster |
| Cluster shape | round (equal spread) | **elliptical**, any orientation and size |
| Output | labels | labels + `predict_proba` + a full density model |

K-Means is the special case of equal, spherical clusters with hard assignments.

## Expectation-Maximization

Alternate until the likelihood stops improving:

- **E-step:** each point's *responsibility* for component `k` — the posterior `weight_k · N(x | mean_k, cov_k) / p(x)`.
- **M-step:** refit each component to the data, weighting every point by its responsibility: new weight = share of responsibility, new mean = weighted average, new covariance = weighted scatter.

Each iteration provably never lowers the log-likelihood (`log_likelihood_history_` makes that visible), but EM only finds a *local* optimum, so initialization matters — here k-means++ seeds plus a few Lloyd steps. Densities are computed in log-space via Cholesky factors; `reg_covar` keeps covariances invertible.

**Choosing `k`:** `bic(X)` penalizes extra parameters — fit several `k` and take the lowest BIC.

## Usage

```python
# (the class lives in notebook.ipynb)

gmm = GaussianMixture(n_components=3).fit(X)
gmm.means_, gmm.covariances_, gmm.weights_
gmm.predict_proba(X)     # soft assignments
gmm.bic(X)
```
