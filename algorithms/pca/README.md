# PCA (Principal Component Analysis)

Finds the directions of maximum variance by eigendecomposing the covariance matrix of the centered data, then projects onto the top `n_components` eigenvectors.

Note: eigenvector sign is mathematically arbitrary — this implementation's components may be flipped relative to scikit-learn's, but explained variance and projection magnitudes match.

## Usage

```python
from model import PCA

model = PCA(n_components=2).fit(X)
X_reduced = model.transform(X)
print(model.explained_variance_ratio_)
```

## Run

```bash
python test_model.py
python demo.py
```
