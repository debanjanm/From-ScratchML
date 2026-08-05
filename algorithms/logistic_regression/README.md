# Logistic Regression

Binary classifier: `p(y=1|x) = sigmoid(Xw + b)`, trained by batch gradient descent minimizing binary cross-entropy.

## Usage

```python
from model import LogisticRegression

model = LogisticRegression(lr=0.1, n_iters=2000).fit(X, y)
probs = model.predict_proba(X_test)
labels = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
