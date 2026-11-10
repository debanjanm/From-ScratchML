# Random Forest

Ensemble of decision trees. Each tree is trained on a bootstrap sample of the rows and, at every split, considers only a random subset of features (`sqrt(n_features)` by default). Prediction is a majority vote across trees. Averaging many decorrelated, overfit trees cuts variance.

Self-contained: includes its own small tree (the plain one is in `10_decision_tree`).

## Usage

```python
from model import RandomForestClassifier

model = RandomForestClassifier(n_estimators=25, max_depth=8).fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
