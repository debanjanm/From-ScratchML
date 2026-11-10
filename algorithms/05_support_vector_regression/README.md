# Support Vector Regression (linear)

SVM idea applied to regression. Fit a line and surround it with a tube of half-width `epsilon`:

- points **inside** the tube cost nothing
- points **outside** cost their distance beyond the tube (linear, not squared)

```
lam/2 * ||w||^2  +  mean( max(0, |y − (w·x + b)| − epsilon) )
```

Only points on or outside the tube (the support vectors) affect the fit. Consequences: it ignores small noise, and a few extreme outliers pull far less than in least squares. `lam` plays the role of `1 / (C·n)` in scikit-learn's `SVR`.

Linear kernel only here (see `11_svm` for the classifier). Scale features and `y`, since `epsilon` is measured in `y`'s units.

## Usage

```python
from model import LinearSVR

model = LinearSVR(epsilon=0.1, lam=0.001).fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
