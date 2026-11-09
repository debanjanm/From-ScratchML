import numpy as np
from model import LassoRegression, RidgeRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge

# 10 features, only 4 actually matter
X, y = make_regression(n_samples=200, n_features=10, n_informative=4, noise=5.0, random_state=0)

ridge_ours = RidgeRegression(alpha=10.0).fit(X, y)
ridge_theirs = Ridge(alpha=10.0).fit(X, y)
lasso_ours = LassoRegression(alpha=1.0).fit(X, y)
lasso_theirs = Lasso(alpha=1.0, tol=1e-10, max_iter=100000).fit(X, y)

np.set_printoptions(precision=2, suppress=True, linewidth=120)
print("ridge  ours  :", ridge_ours.weights)
print("ridge  sklearn:", ridge_theirs.coef_)
print("lasso  ours  :", lasso_ours.weights)
print("lasso  sklearn:", lasso_theirs.coef_)
print("lasso zeroed %d of 10 features" % (lasso_ours.weights == 0).sum())

assert np.allclose(ridge_ours.weights, ridge_theirs.coef_, atol=1e-6)
assert np.allclose(ridge_ours.bias, ridge_theirs.intercept_, atol=1e-6)
assert np.allclose(lasso_ours.weights, lasso_theirs.coef_, atol=1e-4)
assert np.allclose(lasso_ours.bias, lasso_theirs.intercept_, atol=1e-4)
