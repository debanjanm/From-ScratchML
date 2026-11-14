# Gaussian Naive Bayes

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Assumes each feature is normally distributed within a class and independent given the class. Fits per-class mean/variance, predicts via Bayes' rule using log-probabilities for numerical stability.

## Usage

```python
# (the class lives in notebook.ipynb)

model = GaussianNaiveBayes().fit(X_train, y_train)
preds = model.predict(X_test)
```
