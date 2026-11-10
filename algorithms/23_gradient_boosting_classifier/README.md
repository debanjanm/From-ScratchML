# Gradient Boosting Classifier

`22_gradient_boosting` fit residuals `y − F(x)` for regression. Classification uses the same idea with a different loss:

1. Keep one raw score `F_k(x)` per class; probabilities are `softmax(F)`.
2. Each round, for every class `k`, the **negative gradient of the log loss** is the residual `1[y = k] − p_k` — "how much more probability should class k have got".
3. Fit a small regression tree to that residual and add it: `F_k += learning_rate · tree_k(x)`.

**Newton leaves.** A plain average of residuals is only a first-order step. Each leaf instead outputs `(K−1)/K · Σr / Σp(1−p)`, using the second derivative of the loss (the same trick XGBoost builds on, see `24_xgboost`). This converges faster and is what scikit-learn does.

Start point is the class priors as log-odds. Cost: `n_estimators × n_classes` trees. Works for any number of classes, any label type; `loss_history_` records the training log loss each round.

## Usage

```python
from model import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
