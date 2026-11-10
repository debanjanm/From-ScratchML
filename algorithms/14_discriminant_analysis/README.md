# Linear and Quadratic Discriminant Analysis

Generative classifiers: model each class as a **multivariate Gaussian** (its own mean), then classify by Bayes' rule — pick the class with the highest posterior. Unlike Gaussian Naive Bayes (`12_naive_bayes`), features are **not** assumed independent: the full covariance matrix is used, so correlated features are handled properly.

| | Covariance | Decision boundary | Use when |
|---|---|---|---|
| **LDA** | one matrix shared by all classes (pooled) | straight lines | few samples, classes have similar spread |
| **QDA** | one matrix per class | curves (quadratic) | plenty of samples, classes differ in spread |

LDA has fewer parameters to estimate (more stable); QDA is more flexible but needs enough samples per class — `reg` adds `reg·I` to each covariance when a class is small or features are collinear.

Scores are log-posteriors up to a constant; the class priors come from the training class frequencies. (LDA is also a classic supervised dimensionality-reduction tool — that projection isn't implemented here, only the classifier.)

## Usage

```python
from model import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis(reg=1e-6).fit(X_train, y_train)
preds = lda.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
