# LDA Projection (Fisher's Linear Discriminant)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

PCA (`34_pca`) keeps the directions of **largest variance** and never looks at the labels — so it can happily keep a direction that is all noise. LDA is the **supervised** alternative: it picks the directions that best **separate the classes**.

(`14_discriminant_analysis` uses the same maths as a classifier. This folder uses it to *reduce dimensions*.)

## How it works

1. Within-class scatter `Sw = Σ_c Σ_{x∈c} (x − μ_c)(x − μ_c)ᵀ` — how spread out each class is.
2. Between-class scatter `Sb = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ` — how far apart the class means are.
3. Find directions `w` maximising the **Fisher ratio** `wᵀSb w / wᵀSw w`. That is the generalized eigenproblem `Sb w = λ Sw w`.
4. To solve it with a plain symmetric eigensolver: `Sw = L Lᵀ` (Cholesky), eigendecompose `L⁻¹ Sb L⁻ᵀ`, map back with `w = L⁻ᵀ v`. Each `w` is scaled so `wᵀSw w = 1`.
5. Keep the top eigenvectors. **At most `C − 1`** are useful (`Sb` has rank ≤ `C − 1`) — 3 classes give a 2-D view.

The eigenvalue `λ` is the Fisher ratio that direction achieves, so `eigenvalues_` tells you how much separation each axis carries.

Use it when you have labels and want a small, class-aware embedding (visualisation, preprocessing before kNN). Assumes classes are roughly Gaussian with similar covariance; needs more samples than features for a stable `Sw`.

## Usage

```python
# (the class lives in notebook.ipynb)

lda = LDAProjection(n_components=2).fit(X, y)
Z = lda.transform(X)
lda.eigenvalues_, lda.explained_variance_ratio_
```
