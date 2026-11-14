# Softmax Regression

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Logistic regression (`09_logistic_regression`) generalized to **any number of classes**. Instead of one sigmoid, compute a score per class and normalize with softmax:

```
P(class k | x) = exp(x·W_k + b_k) / Σ_j exp(x·W_j + b_j)
```

Trained by gradient descent on cross-entropy. The gradient has the same neat form as the binary case: `Xᵀ(P − Y) / n`, where `Y` is the one-hot target. With 2 classes it reduces to ordinary logistic regression.

This is also exactly the output layer of `26_neural_network`.

Softmax subtracts the row maximum before `exp` so large scores can't overflow. Scale features first.

## Usage

```python
# (the class lives in notebook.ipynb)

model = SoftmaxRegression(lr=0.1, n_iters=1000).fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)    # one column per class
```
