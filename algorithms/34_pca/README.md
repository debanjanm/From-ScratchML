# PCA (Principal Component Analysis)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Finds the directions of maximum variance by eigendecomposing the covariance matrix of the centered data, then projects onto the top `n_components` eigenvectors.

Note: eigenvector sign is mathematically arbitrary — this implementation's components may be flipped relative to scikit-learn's, but explained variance and projection magnitudes match.

## Usage

```python
# (the class lives in notebook.ipynb)

model = PCA(n_components=2).fit(X)
X_reduced = model.transform(X)
print(model.explained_variance_ratio_)
```
