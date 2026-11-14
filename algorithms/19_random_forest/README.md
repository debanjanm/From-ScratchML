# Random Forest

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Ensemble of decision trees. Each tree is trained on a bootstrap sample of the rows and, at every split, considers only a random subset of features (`sqrt(n_features)` by default). Prediction is a majority vote across trees. Averaging many decorrelated, overfit trees cuts variance.

Self-contained: includes its own small tree (the plain one is in `15_decision_tree`).

## Usage

```python
# (the class lives in notebook.ipynb)

model = RandomForestClassifier(n_estimators=25, max_depth=8).fit(X_train, y_train)
preds = model.predict(X_test)
```
