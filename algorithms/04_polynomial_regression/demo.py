import numpy as np
from model import PolynomialRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(150, 2))
y = 1 + 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 0] ** 2 + X[:, 0] * X[:, 1] + rng.normal(scale=0.3, size=150)

for degree in (1, 2, 3):
    ours = PolynomialRegression(degree=degree).fit(X, y)
    theirs = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression()).fit(X, y)
    mse = np.mean((ours.predict(X) - y) ** 2)
    print("degree %d: train MSE=%.4f" % (degree, mse))
    assert np.allclose(ours.predict(X), theirs.predict(X), atol=1e-6)

try:
    import matplotlib.pyplot as plt

    x1 = rng.uniform(-3, 3, 100)
    x1.sort()
    y1 = 0.5 * x1**3 - x1 + rng.normal(scale=1.0, size=100)
    plt.scatter(x1, y1, s=10, alpha=0.5)
    for degree, color in ((1, "gray"), (3, "red")):
        m = PolynomialRegression(degree).fit(x1[:, None], y1)
        plt.plot(x1, m.predict(x1[:, None]), color=color, label="degree %d" % degree)
    plt.legend()
    plt.title("Polynomial Regression (from scratch)")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
