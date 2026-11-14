# Voting and Stacking

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Where bagging and boosting build many copies of *one kind* of model, these two combine **different kinds** of models (logistic regression + KNN + a tree, say). Their errors tend to differ, so combining them beats each one alone.

The base learners are **plugged in from outside** — anything with `fit` / `predict` (and optionally `predict_proba`), from this repo or elsewhere.

## Voting

| | How | Notes |
|---|---|---|
| `hard` | each model votes for a class, majority wins | ignores confidence |
| `soft` | average the predicted probabilities, take the max | one very confident model can outweigh two unsure ones |

## Stacking

Learn *how* to combine, instead of a fixed rule:

1. Split into `cv` folds. For each fold, train every base model on the *other* folds and predict this fold → **out-of-fold** predictions.
2. Train a **meta-model** on `[those predictions → true labels]`.
3. Refit the base models on all data. At prediction time: base predictions → meta-model.

Step 1 is the crucial part: a base model's predictions on rows it was trained on look unrealistically good (a memorizing model looks perfect), so a meta-model trained on them would over-trust it. Out-of-fold predictions are honest. The test suite checks this by spying on the training-set sizes.

Default meta-model is a built-in softmax regression; pass `final_estimator=` to use another.

## Usage

```python
# (the class lives in notebook.ipynb)

base = [("a", ModelA()), ("b", ModelB()), ("c", ModelC())]
vote = VotingClassifier(base, voting="soft").fit(X_train, y_train)
stack = StackingClassifier(base, cv=5).fit(X_train, y_train)
preds = stack.predict(X_test)
```
