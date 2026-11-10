# Extra Trees (Extremely Randomized Trees)

A Random Forest (`19_random_forest`) with two changes that add even more randomness:

| | Random Forest | Extra Trees |
|---|---|---|
| Training rows per tree | bootstrap sample | the **whole** set (by default) |
| Candidate features per split | random subset | random subset |
| Split threshold | **best** cut, found by scanning | **random** cut, uniform in the feature's range |

At each node, a random threshold is drawn for each candidate feature and the best-scoring of those random splits wins. Because nothing is sorted or scanned for the optimum, each tree is much cheaper to build, and the extra randomness decorrelates the trees further — lowering ensemble variance at the cost of slightly higher bias per tree. Averaging many trees makes up for the weaker individual splits.

## Usage

```python
from model import ExtraTreesClassifier

model = ExtraTreesClassifier(n_estimators=100).fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
