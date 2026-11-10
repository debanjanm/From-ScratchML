# t-SNE

A non-linear way to **look at** high-dimensional data in 2-D. PCA (`34_pca`) preserves large-scale variance and flattens curved structure; t-SNE preserves **who is near whom**, so clusters (like the ten digit classes) separate cleanly.

## How it works

1. **Neighbour probabilities in the original space.** For each point `i`, put a Gaussian over the other points and read off `p(j|i)`. The Gaussian's width is found by binary search so that the distribution has entropy `log(perplexity)` — `perplexity` ≈ effective number of neighbours (try 5–50). Symmetrize: `P = (p(j|i) + p(i|j)) / 2n`.
2. **Neighbour probabilities in the embedding.** `Q[i,j] ∝ 1 / (1 + ‖yᵢ − yⱼ‖²)` — a Student-t (heavy-tailed) kernel. The heavy tail lets moderately-distant points sit very far apart, which fixes the "crowding problem" of squeezing high-dim neighbourhoods into 2-D.
3. **Minimise `KL(P ‖ Q)`** over the 2-D positions by gradient descent:
   `∂KL/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ) / (1 + ‖yᵢ − yⱼ‖²)`
   with momentum (0.5, then 0.8), per-parameter gains, and **early exaggeration** (`P × 12` for the first 250 steps) so clusters form tight and well apart.

The test checks the gradient against finite differences and that the binary search really hits the requested perplexity. In the demo on 600 digits, KL and trustworthiness match scikit-learn's exact t-SNE (KL 0.377 vs 0.383), while a linear PCA map keeps far fewer neighbours.

## Read the picture carefully

- Cluster **sizes** and **distances between** clusters are not meaningful.
- No `transform()` for new points — it embeds exactly the data you give it.
- Different seeds / perplexities give different pictures. Treat it as exploration, not as features for a model.

`ponytail:` exact version, `O(n²)` per iteration, fine up to a few thousand points. Barnes-Hut or FFT variants are the upgrade for more.

## Usage

```python
from model import TSNE

tsne = TSNE(perplexity=30, n_iter=1000, random_state=0)
Y = tsne.fit_transform(X)      # n x 2
tsne.kl_divergence_            # final objective, lower is better
```

## Run

```bash
python test_model.py
python demo.py      # ~15 s: includes scikit-learn's slow exact t-SNE for comparison
```
