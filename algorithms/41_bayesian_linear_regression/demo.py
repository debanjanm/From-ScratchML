import numpy as np
from model import BayesianLinearRegression
from sklearn.linear_model import BayesianRidge

zero_prior = dict(alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0, tol=1e-10, max_iter=1000)  # plain evidence maximisation

# 1) same learned precisions and weights as scikit-learn's BayesianRidge
rng = np.random.default_rng(0)
X = rng.normal(size=(150, 6))
w = np.array([3.0, 0.0, -2.0, 0.0, 0.5, 0.0])
y = X @ w + 1.0 + rng.normal(scale=0.5, size=150)

ours = BayesianLinearRegression().fit(X, y)
theirs = BayesianRidge(**zero_prior).fit(X, y)
print("noise precision alpha: ours=%.4f sklearn=%.4f" % (ours.alpha_, theirs.alpha_))
print("weight precision lambda: ours=%.4f sklearn=%.4f" % (ours.lambda_, theirs.lambda_))
assert np.isclose(ours.alpha_, theirs.alpha_, rtol=1e-4) and np.isclose(ours.lambda_, theirs.lambda_, rtol=1e-4)
assert np.allclose(ours.coef_, theirs.coef_, atol=1e-6) and np.isclose(ours.intercept_, theirs.intercept_, atol=1e-6)
print("coefficients:", ours.coef_.round(3), "(true", w, ")")

# 2) predictive standard deviation (compared on centred data: sklearn ignores the centring when computing it)
Xc, yc = X - X.mean(0), y - y.mean()
ours0 = BayesianLinearRegression(fit_intercept=False).fit(Xc, yc)
theirs0 = BayesianRidge(fit_intercept=False, **zero_prior).fit(Xc, yc)
assert np.allclose(ours0.predict(Xc, return_std=True)[1], theirs0.predict(Xc, return_std=True)[1], atol=1e-6)
print("predictive std matches sklearn")

# 3) the point of being Bayesian: honest error bars that widen away from the data (cubic fit)
x = rng.uniform(-1, 1, size=40)
t = np.sin(2.5 * x) + rng.normal(scale=0.1, size=40)
poly = lambda v: np.column_stack([np.asarray(v) ** p for p in (1, 2, 3)])
model = BayesianLinearRegression().fit(poly(x), t)
for point in (0.0, 0.9, 1.5, 3.0):
    mean, std = model.predict(poly([point]), return_std=True)
    print("x=%.1f  prediction %6.2f +/- %.2f   (truth %5.2f)" % (point, mean[0], 1.96 * std[0], np.sin(2.5 * point)))
