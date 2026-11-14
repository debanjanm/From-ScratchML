# Interview Guide

How to use this repo for ML interviews: how to think, how to remember 42 algorithms, and the numpy moves that keep repeating.

---

## 1. How to think in an interview

### The loop (use it for every algorithm question)

1. **Clarify.** Inputs and outputs? Labels or not? How big is `n` and `d`? Is a numpy-only answer expected, or can you name the sklearn call? (30 seconds, and it shows judgment.)
2. **Say the model in three lines.** Every algorithm is *(model, loss, solver)*:
   - **Model** — what function maps `x` to a prediction?
   - **Loss / objective** — what does "good" mean?
   - **Solver** — closed form, gradient descent, greedy search, EM, or eigendecomposition?
3. **Derive the one formula that matters.** Don't recite. Derive the gradient or the update from the loss in three lines.
4. **Write the skeleton first.** `__init__` → `fit` → `predict`, shapes in comments. Fill in the math after.
5. **Check it.** Say how you would know it is right: gradient check, compare to a brute-force version, loss goes down, a tiny example by hand. The *Tests* section of each notebook shows the check for that algorithm.
6. **Finish with trade-offs.** Complexity, assumptions, when it breaks, what you would use instead. This is where senior answers separate from junior ones.

### The questions interviewers keep asking

| Question | What they want |
|---|---|
| Time / memory complexity of `fit` and `predict`? | `O(·)` in `n`, `d`, `k`, iterations. Name the bottleneck. |
| What are the assumptions? | Linearity, independence (NB), Gaussian classes (LDA), round clusters (K-Means)… |
| How does it overfit and how do you control it? | Regularization, depth, `k`, early stopping, more data. |
| Bias vs variance? | Bagging cuts **variance**; boosting cuts **bias**; deep trees = low bias / high variance; big `k` in kNN = high bias. |
| Do you need to scale features? | Yes: kNN, SVM, K-Means, PCA, ridge/lasso, gradient descent. No: trees, forests, boosting, NB. |
| What if it doesn't converge / is unstable? | Learning rate, scaling, numerical stability (see §4), restarts (EM, K-Means). |
| Why not just use X? | Name the case where it wins *and* the case where it loses. |

### If you get stuck

- Say what you know ("it is a greedy split search over thresholds") before what you don't. Silence is worse than a partial answer.
- Write the **`fit`/`predict` skeleton and the shapes** and keep going; partial credit is real.
- Simplify: solve 1 feature, 2 classes, 1 iteration first. Then generalize.

---

## 2. How to remember 42 algorithms

**Do not memorize 42 things. Memorize ~12 ideas and the one twist that makes each algorithm different.**

### The 12 ideas

| Idea | One sentence | Algorithms in this repo |
|---|---|---|
| **Fit a line, minimize squared error** | Closed form `w = (XᵀX)⁻¹Xᵀy` | `01` Linear · `04` Polynomial (just add features) |
| **Add a penalty** | Ridge `+λ‖w‖²` (shrinks), Lasso `+λ‖w‖₁` (zeros out), Elastic Net = both | `02`, `03` |
| **Squash a linear score** | sigmoid / softmax → probabilities; gradient is always `Xᵀ(pred − y)/n` | `09` Logistic · `10` Softmax · `26` Neural net |
| **Max-margin** | Hinge loss; only support vectors matter; kernels = similarity instead of features | `05` SVR · `16` SVM · `17` Kernel SVM |
| **Bayes' rule with a simplifying assumption** | `argmax log prior + Σ log likelihood` | `12`, `13` NB · `14` LDA/QDA |
| **Lazy: remember the data** | Predict from the `k` nearest points | `11` kNN · `31` Mean Shift |
| **Greedy tree splits** | Pick the (feature, threshold) with best impurity drop; sort + cumulative sum | `07`, `15` Trees |
| **Many models, average** | Bootstrap + average → lower **variance** | `18` Bagging · `19` RF · `20` Extra Trees |
| **Many models, in sequence** | Each fixes the last one's mistakes → lower **bias** | `21` AdaBoost · `22`/`23` GB · `24` XGBoost |
| **Alternate two steps** | Assign ↔ update, E ↔ M. Monotone, local optimum | `27` K-Means · `28` K-Medoids · `29` GMM · `42` HMM (Baum-Welch) |
| **Eigen / SVD of a matrix** | Directions of largest variance, or smallest graph cut | `34` PCA · `35` SVD · `36` Kernel PCA · `37` LDA proj · `33` Spectral |
| **Score by neighbourhood / isolation** | Density vs. neighbours; how fast a point is isolated | `32` DBSCAN · `39` IsoForest · `40` LOF · `30` Hierarchical |

Left over: `06` GP and `41` Bayesian LR (Gaussian posterior over functions/weights), `08` Perceptron (update only on mistakes), `25` Voting/Stacking (combine models), `38` t-SNE (match neighbour probabilities).

### Memory techniques that work

1. **Compress each algorithm to one line, then expand from the loss.** Example for logistic regression: *"sigmoid of linear, cross-entropy loss, gradient `Xᵀ(p − y)/n`."* If you can say that, you can derive and code the rest.
2. **Learn the twists, not the algorithms.** Ridge vs. lasso differ by *one term*. Logistic vs. softmax differ by *sigmoid → softmax*. RF vs. bagging: *random feature subset per split*. Extra Trees vs. RF: *random thresholds*. AdaBoost vs. GB: *reweight samples* vs. *fit residuals*.
3. **Know one failure case per algorithm.** K-Means on moons, NB with correlated features, kNN in high dimensions, PCA when signal is non-linear, unscaled SVM. Each notebook's *See it work* section shows the contrast.
4. **Rewrite from a blank file.** Reading is not remembering. Pick an algorithm, open an empty file, time 20 minutes, then compare against the *Implementation* cell of the notebook. Repeat in 3 days, then in 10.
5. **Use the tests as a checklist.** The *Tests* cells list what must be true of the algorithm (loss decreases, probabilities sum to 1, gradient matches finite differences). Knowing the invariants is knowing the algorithm.
6. **Compare in pairs.** Doing two similar things side by side (Ridge/Lasso, RF/Extra Trees, K-Means/K-Medoids, PCA/Kernel PCA/LDA) sticks better than studying each alone.

### The "must write from memory" list (highest frequency in interviews)

Linear regression (closed form + gradient descent) · logistic regression · k-means · kNN · PCA · decision tree split · naive Bayes · softmax · random forest (bagging + feature subsampling) · gradient boosting for squared loss · simple MLP with backprop.

If you can do these eleven blind, you are covered for most coding rounds.

---

## 3. The five templates that cover most of the repo

**A. Estimator skeleton**

```python
class Model:
    def __init__(self, ...): ...                  # hyperparameters only
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        ...                                       # learn; store with trailing underscore: self.coef_
        return self
    def predict(self, X): ...
```

**B. Gradient descent loop** (`01`, `09`, `10`, `16`, `26`)

```python
for _ in range(n_iters):
    pred = f(X, w)                  # forward
    grad = X.T @ (pred - y) / n     # backward  (linear, logistic and softmax all look like this)
    w -= lr * grad                  # step
```

**C. Greedy split** (`07`, `15`, `18`–`24`): for each feature, `argsort`, cumulative sums for the left side, gain vector for *every* threshold at once, `argmax`, recurse.

**D. Alternating (EM-style)** (`27`, `29`, `42`): loop { assign / E-step (soft or hard) → update / M-step }; track the objective; restart with different seeds.

**E. Kernel / affinity matrix** (`06`, `17`, `33`, `36`): build `K` (n×n) from pairwise distances → `solve` / `eigh` on it → predict with the kernel row of new points.

---

## 4. NumPy tricks that repeat

Where each appears in this repo is in the right column; open that file when you want a worked example.

| Trick | Snippet | Used in |
|---|---|---|
| **Bias column** | `Xb = np.hstack([np.ones((len(X), 1)), X])` | `01` (logistic/softmax keep `w` and `b` separate) |
| **Center data** | `Xc = X - X.mean(axis=0)`; cov = `Xc.T @ Xc / (n - 1)` | `34`, `37` (SVD `35` deliberately skips it) |
| **Gradient for all GLMs** (up to a constant) | `X.T @ (pred - y) / n` | `01`, `09`, `10`, `26` |
| **Pairwise squared distances** (no loops) | `d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T`; then `np.maximum(d2, 0)` | `36`, `38`, `40` (kNN `11` does one query at a time: `np.linalg.norm(X_train - x, axis=1)`) |
| **Same, by broadcasting** (simpler, `O(nmd)` memory) | `((A[:, None] - B[None]) ** 2).sum(-1)` | `27`, `33` |
| **k nearest** | `np.argsort(D, axis=1)[:, :k]` (use `np.fill_diagonal(D, np.inf)` to exclude self) | `11`, `33`, `40` |
| **Stable softmax** | `e = np.exp(Z - Z.max(1, keepdims=True)); e / e.sum(1, keepdims=True)` | `10`, `26` |
| **Log-sum-exp** | `m = a.max(1, keepdims=True); m + np.log(np.exp(a - m).sum(1, keepdims=True))` | `29` |
| **One-hot** | `np.eye(k)[y_enc]` | `10`, `26` |
| **Encode labels** | `classes, y_enc = np.unique(y, return_inverse=True)` | most classifiers |
| **Class counts / priors** | `np.bincount(y_enc) / len(y_enc)` | `13`, `14`, `15` |
| **Per-class stats** | `np.array([X[y_enc == c].mean(0) for c in range(k)])` | `12`, `14`, `37` |
| **Best split for *all* thresholds at once** | sort, `np.cumsum(ys)[:-1]`, gain vector, `argmax`; mask ties with `np.where(xs[1:] != xs[:-1], gain, -np.inf)` | `07`, `21`, `22` |
| **Solve, don't invert** | `np.linalg.solve(A, b)` (never `inv(A) @ b`) | `02`, `06` |
| **SPD matrices** | `np.linalg.cholesky(K)`, then triangular solves | `06`, `29`, `37` |
| **Eigen, symmetric** | `vals, vecs = np.linalg.eigh(C)`; it returns **ascending**, so `[::-1]` or `argsort(vals)[::-1]` for the top-k | `34`, `35`, `36`, `33` |
| **Bootstrap** | `idx = rng.integers(0, n, n)`; samples never drawn are the out-of-bag set | `18`, `19` |
| **Seeded randomness** | `rng = np.random.default_rng(seed)` (don't touch the global state) | everywhere |
| **Conditional values** | `np.where(cond, a, b)` · boolean masks `X[mask]` | trees, SVM |
| **Hinge violators / ReLU** | `y * score < 1` (hinge gradient is nonzero only there) · `np.maximum(Z, 0)` | `16`, `26` |
| **Row-normalize** | `M / M.sum(axis=1, keepdims=True)` | `10`, `42` |
| **Outer product** | `a[:, None] * b[None, :]` (or `diff[:, None] @ diff[None, :]`) | `37`, `42` |
| **Soft-threshold** (lasso) | `np.sign(z) * np.maximum(np.abs(z) - lam, 0)` | `02`, `03` |

### Shape habits

- Write shapes in comments: `# X: (n, d)  W: (d, k)  Z: (n, k)`. Most bugs are shape bugs.
- `keepdims=True` is how you make a reduction broadcast back correctly. If a division gives a wrong-shaped result, you forgot it.
- `X.T @ y` with 1-D `y` returns 1-D; `y.reshape(-1, 1)` makes it a column. Pick one and stay with it.
- Check `(n, 1)` vs `(n,)` first when broadcasting gives a surprise `(n, n)`.

### Numerical-stability checklist (interviewers love this)

- Softmax / log-sum-exp: subtract the max.
- `log(0)`: `np.log(p + 1e-12)` or `np.maximum(p, 1e-12)`.
- Sigmoid with large `|z|`: clip `z` to ±500 or use the two-branch form.
- Probabilities of long sequences: work in log space or rescale each step (HMM `42`).
- Covariance / Gram matrices: add a tiny ridge `+ eps * I` before inverting or Cholesky.
- Never `inv(XᵀX)` for ill-conditioned data: use `solve`, `pinv`, `lstsq`, or SVD.
- Standardize features before any distance-based or gradient-based method.
- Euclidean distance via the `‖a‖² + ‖b‖² − 2a·b` trick can go slightly negative: `np.maximum(d2, 0)` before `sqrt`.

---

## 5. One-line recall sheet

| Algorithm | Recall line |
|---|---|
| Linear | `w = solve(XᵀX, Xᵀy)`; loss = MSE |
| Ridge / Lasso | `solve(XᵀX + λI, Xᵀy)` / coordinate descent with soft-threshold |
| Logistic | `p = σ(Xw)`; grad `Xᵀ(p − y)/n`; cross-entropy |
| Softmax | same with `Y` one-hot and softmax rows |
| Perceptron | on a mistake: `w += y·x` |
| SVM | hinge loss + `½‖w‖²`; kernel version = SMO on the dual |
| kNN | `argsort` distances, majority/mean of `k` labels |
| Naive Bayes | `argmax log π_c + Σ_j log p(x_j│c)` |
| LDA | shared covariance → linear boundary; QDA = per-class covariance |
| Decision tree | best split maximizes impurity drop (Gini `1 − Σp²`, or variance reduction) |
| Random Forest | bootstrap rows + random feature subset per split; average/vote |
| AdaBoost | weak learner `h`, weight `α = ½ ln((1 − ε)/ε)`, upweight mistakes |
| Gradient Boosting | each tree fits the **negative gradient** (residuals for MSE), scaled by learning rate |
| XGBoost | second-order: leaf `w = −G/(H + λ)`; gain `½(G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)) − γ` |
| K-Means | assign nearest centroid → recompute means; inertia never increases |
| GMM | E: responsibilities; M: weighted means/covs/weights; likelihood never decreases |
| DBSCAN | core points have ≥ `min_samples` neighbours in `eps`; clusters = connected cores |
| Hierarchical | merge closest pair; linkage = how to measure cluster distance |
| Spectral | affinity graph → top eigenvectors of `D^-½ W D^-½` → K-Means on rows |
| PCA | eigenvectors of the covariance of centered data (or SVD of `Xc`) |
| t-SNE | match neighbour probabilities `P` (Gaussian) to `Q` (Student-t) by minimizing KL |
| Isolation Forest | random cuts; short average path = anomaly |
| LOF | density of a point ÷ density of its neighbours; ≫ 1 = outlier |
| GP regression | mean `k*ᵀ(K + σ²I)⁻¹y`, variance `k** − k*ᵀ(K + σ²I)⁻¹k*` |
| HMM | forward = `Σ` over paths, Viterbi = `max`; Baum-Welch = EM on both |

---

## 6. A practice plan

- **Week 1 — the eleven blind-write list** (§2). One per day, 20 minutes each, blank file, then diff against the repo.
- **Week 2 — twists and comparisons.** Pairs: Ridge/Lasso, Bagging/RF/Extra Trees, AdaBoost/GB, K-Means/GMM/DBSCAN, PCA/LDA/Kernel PCA.
- **Week 3 — explain out loud.** For each algorithm give the 3-line version (model, loss, solver), one failure case, the complexity. Record yourself or teach a friend.
- **Always:** after writing something, ask *"how would I know this is right?"* and write that check. That habit is what these folders' tests are modelling.
