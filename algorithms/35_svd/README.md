# Truncated SVD

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Every matrix factors as `X = U · S · Vᵀ`: `V` holds orthonormal directions in feature space, `S` the strength of each, `U` the coordinates of each sample. Keep only the `k` largest singular values and you get the **best rank-`k` approximation of `X`** (Eckart–Young): no other rank-`k` matrix is closer.

## How it works

1. Form the Gram matrix `XᵀX`. Its eigenvalues are the squared singular values `s²`, its eigenvectors the right singular vectors `V`.
2. Sort descending, keep the top `k`: `components_` is `Vₖᵀ`, `singular_values_` is `sₖ`.
3. `transform(X) = X · Vₖ` (this is `Uₖ Sₖ`); `inverse_transform(Z) = Z · Vₖᵀ` is the rank-`k` reconstruction.
4. Reconstruction error is exactly `Σ s²` over the dropped values — the test checks this.

## SVD vs PCA

PCA = SVD of the **centered** data. `TruncatedSVD` skips the centering, so it works on data where zero means "nothing there" (word counts, tf-idf, ratings) and the mean is not a meaningful origin. If your data has a big offset, the first SVD component just points at the mean — that is what centering in PCA removes.

Common uses: latent semantic analysis (documents × terms), image compression, denoising, matrix completion building block.

`ponytail:` this builds `XᵀX`, which squares the condition number and costs `O(n·d²)`. For big or ill-conditioned matrices use `np.linalg.svd` or a randomized/Lanczos solver.

## Usage

```python
# (the class lives in notebook.ipynb)

svd = TruncatedSVD(n_components=10).fit(X)
Z = svd.transform(X)               # n x 10
X_approx = svd.inverse_transform(Z)  # rank-10 reconstruction
svd.singular_values_, svd.explained_variance_ratio_
```
