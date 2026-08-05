# Gaussian Naive Bayes

Assumes each feature is normally distributed within a class and independent given the class. Fits per-class mean/variance, predicts via Bayes' rule using log-probabilities for numerical stability.

## Usage

```python
from model import GaussianNaiveBayes

model = GaussianNaiveBayes().fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
