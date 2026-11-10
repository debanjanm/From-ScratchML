# Kernel PCA

PCA (`34_pca`) can only find **straight** directions, so it cannot untangle two concentric rings: both have the same mean and the same variance in every direction. Kernel PCA does PCA in a **curved, high-dimensional feature space** `φ(x)` — without ever computing `φ`, thanks to the kernel trick.

## How it works

1. Pick a kernel `k(a, b) = φ(a)·φ(b)` and build the `n × n` matrix `K`:
   - `linear`: `a·b` (gives ordinary PCA back)
   - `poly`: `(γ·a·b + c₀)^d`
   - `rbf`: `exp(−γ‖a − b‖²)`
2. **Center** it so the implicit features have zero mean: `Kc = K − 1K − K1 + 1K1`.
3. Eigendecompose `Kc`. The score of training point `i` on component `j` is `√λⱼ · vⱼ[i]`.
4. A **new** point is scored with its kernel row against the training set, centered the same way, times `vⱼ/√λⱼ`.

On two rings an RBF kernel makes points on the same ring similar, so the first component alone splits inner from outer — the demo shows a single threshold separating them (≥ 99%) where PCA gets ~50–70%.

Cost: stores and eigendecomposes the full `n × n` matrix (`O(n²)` memory, `O(n³)` time), so it suits thousands of points, not millions. `γ` matters a lot — too large and every point is only similar to itself.

## Usage

```python
from model import KernelPCA

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=10).fit(X)
Z = kpca.transform(X)
Z_new = kpca.transform(X_new)   # unseen points
```

## Run

```bash
python test_model.py
python demo.py
```
