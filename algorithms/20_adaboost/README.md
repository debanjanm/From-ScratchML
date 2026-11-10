# AdaBoost

Boosting with decision **stumps** (one-split trees), each barely better than guessing. Rounds are sequential:

1. Train a stump on the weighted data; compute its weighted error `err`.
2. Give it a vote strength `alpha = ½ ln((1 − err) / err)` — more accurate stumps count more.
3. Re-weight samples: `w *= exp(−alpha · y · pred)`. Misclassified samples grow, so the next stump focuses on them.

Prediction is `sign(Σ alpha · stump(x))`.

Compare `19_gradient_boosting` (fits residuals) and `18_random_forest` (independent trees, parallel). Binary classification only; labels can be any two values.

## Usage

```python
from model import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
preds = model.predict(X_test)
scores = model.decision_function(X_test)   # signed vote total
```

## Run

```bash
python test_model.py
python demo.py
```
