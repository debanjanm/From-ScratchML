# Decision Tree

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

CART classifier: recursively splits on the feature/threshold that maximizes Gini-impurity reduction, until `max_depth`, a pure leaf, or `min_samples_split` stops growth.

## Usage

```python
# (the class lives in notebook.ipynb)

model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
preds = model.predict(X_test)
```
