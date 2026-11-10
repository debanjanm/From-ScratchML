# Bagging

**B**ootstrap **agg**regat**ing**. Train many models, each on a different random resample of the training set (drawn *with replacement*, so ~63% of the distinct rows appear), then combine by majority vote.

Why it works: a deep decision tree has low bias but **high variance** — change the data slightly and the tree changes a lot. Averaging many such trees cancels the noise while keeping the low bias. Models that are already stable (e.g. linear regression) gain little.

- Works with **any base learner**: pass `make_estimator=lambda: MyModel()`. The default is a CART tree.
- **Out-of-bag score:** each model never saw the ~37% of rows left out of its bootstrap sample, so predicting those rows with only the models that didn't train on them gives a free validation estimate (`oob_score_`) — no hold-out set needed.
- Trees are trained independently, so this parallelizes trivially.

Next steps from here: `19_random_forest` also samples *features* at each split to decorrelate the trees; `20_extra_trees` randomizes the split thresholds too.

## Usage

```python
from model import BaggingClassifier

model = BaggingClassifier(n_estimators=50).fit(X_train, y_train)
preds = model.predict(X_test)
print(model.oob_score_)
```

## Run

```bash
python test_model.py
python demo.py
```
