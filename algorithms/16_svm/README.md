# Support Vector Machine (linear)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Finds the hyperplane `w·x + b = 0` that separates two classes with the widest margin. Trained by subgradient descent on the regularized hinge loss:

```
lam/2 * ||w||^2  +  mean( max(0, 1 - y * (w·x + b)) )
```

Only points inside or beyond the margin (`y * f(x) < 1`) contribute to the gradient — those are the support vectors. `lam` is the regularization strength (large `lam` = wider, softer margin; roughly `1 / C`).

Binary only, linear kernel. Scale features first (the demo uses `StandardScaler`).

## Usage

```python
# (the class lives in notebook.ipynb)

model = LinearSVM(lam=0.01, n_iters=1000).fit(X_train, y_train)
preds = model.predict(X_test)
scores = model.decision_function(X_test)  # signed distance-like score
```
