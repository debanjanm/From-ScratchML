# Neural Network (MLP classifier)

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

Feedforward network with ReLU hidden layers and a softmax output, trained with cross-entropy loss.

**Forward:** `a = relu(a_prev @ W + b)` per hidden layer; the last layer outputs logits, then softmax gives class probabilities.

**Backward (backpropagation):** for softmax + cross-entropy the gradient at the logits is simply `(probs - onehot) / n`. Then, layer by layer from the output back:

```
dW = a_prev.T @ dz
db = dz.sum(0)
dz_prev = (dz @ W.T) * (z_prev > 0)     # through the weights, then the ReLU gate
```

Weights use He initialization and are updated by mini-batch gradient descent. The notebook's tests check the hand-written gradients against finite differences — the standard way to verify backprop.

## Usage

```python
# (the class lives in notebook.ipynb)

net = NeuralNetClassifier(hidden_sizes=(32,), lr=0.1, epochs=50).fit(X_train, y_train)
preds = net.predict(X_test)
probs = net.predict_proba(X_test)
print(net.loss_history_)   # training loss per epoch
```

Scale inputs first (e.g. standardize, or divide pixels by 255).
