# Gradient Boosting (regression)

Builds an additive model `F(x) = mean(y) + lr * sum(tree_m(x))`. Each shallow regression tree is fit to the residuals `y - F(x)` of the ensemble so far, which is the negative gradient of squared error. Small `learning_rate` with many trees generalizes best.

Unlike Random Forest (parallel, variance-reducing), trees here are sequential and bias-reducing.

## Usage

```python
from model import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
preds = model.predict(X_test)
```

## Run

```bash
python test_model.py
python demo.py
```
