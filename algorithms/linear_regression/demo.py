import numpy as np
from model import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = rng.uniform(-10, 10, size=(150, 1))
y = 4.0 * X[:, 0] + 7.0 + rng.normal(scale=1.5, size=150)

ours = LinearRegression(method="normal_equation").fit(X, y)
theirs = SklearnLinearRegression().fit(X, y)

print("ours  : weight=%.4f bias=%.4f" % (ours.weights[0], ours.bias))
print("sklearn: weight=%.4f bias=%.4f" % (theirs.coef_[0], theirs.intercept_))

mse_ours = mean_squared_error(y, ours.predict(X))
mse_theirs = mean_squared_error(y, theirs.predict(X))
print("MSE ours=%.4f sklearn=%.4f" % (mse_ours, mse_theirs))
assert abs(mse_ours - mse_theirs) < 1e-6, "our MSE should match sklearn's"

try:
    import matplotlib.pyplot as plt

    order = np.argsort(X[:, 0])
    plt.scatter(X, y, s=10, alpha=0.5, label="data")
    plt.plot(X[order], ours.predict(X)[order], color="red", label="ours")
    plt.legend()
    plt.title("Linear Regression (from scratch)")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
