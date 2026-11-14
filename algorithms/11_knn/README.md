# K-Nearest Neighbors

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Brute-force classifier: for each query point, find the `k` closest training points (Euclidean distance) and vote on the majority label.

## Usage

```python
# (the class lives in notebook.ipynb)

model = KNNClassifier(k=5).fit(X_train, y_train)
preds = model.predict(X_test)
```
