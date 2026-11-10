import numpy as np
from model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet as SklearnElasticNet

X, y = make_regression(n_samples=200, n_features=10, n_informative=4, noise=5.0, random_state=0)

ours = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X, y)
theirs = SklearnElasticNet(alpha=1.0, l1_ratio=0.5, tol=1e-10, max_iter=100000).fit(X, y)

np.set_printoptions(precision=2, suppress=True, linewidth=120)
print("ours   :", ours.weights)
print("sklearn:", theirs.coef_)
print("zeroed %d of 10 features" % (ours.weights == 0).sum())

assert np.allclose(ours.weights, theirs.coef_, atol=1e-4)
assert np.allclose(ours.bias, theirs.intercept_, atol=1e-4)
