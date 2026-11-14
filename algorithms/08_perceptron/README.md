# Perceptron

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

The simplest neural unit and the ancestor of logistic regression, SVMs and the MLP. It draws a line `w·x + b = 0`; whenever a sample lands on the wrong side, push the line toward it:

```
if y * (w·x + b) <= 0:      # y in {-1, +1}
    w += lr * y * x
    b += lr * y
```

- **Guaranteed to converge** if the classes are linearly separable (the perceptron convergence theorem), after which `errors_` hits 0.
- **Never converges** otherwise — XOR is the classic counter-example, which is what motivated multi-layer networks (`26_neural_network`).
- It finds *a* separating line, not the widest-margin one (that is `16_svm`), and gives no probabilities (that is `09_logistic_regression`).

Binary only; labels can be any two values.

## Usage

```python
# (the class lives in notebook.ipynb)

model = Perceptron(epochs=100).fit(X_train, y_train)
preds = model.predict(X_test)
print(model.errors_)    # mistakes per epoch
```
