# Decision Tree

CART classifier: recursively splits on the feature/threshold that maximizes Gini-impurity reduction, until `max_depth`, a pure leaf, or `min_samples_split` stops growth.

## Usage

```python
from model import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
