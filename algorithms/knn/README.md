# K-Nearest Neighbors

Brute-force classifier: for each query point, find the `k` closest training points (Euclidean distance) and vote on the majority label.

## Usage

```python
from model import KNNClassifier

model = KNNClassifier(k=5).fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
