import numpy as np
from model import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as SklearnGPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 6, size=(25, 1)), axis=0)
y = np.sin(X[:, 0]) + rng.normal(scale=0.1, size=25)
X_test = np.linspace(-1, 7, 80)[:, None]

length_scale, variance, noise = 1.0, 1.0, 0.1
ours = GaussianProcessRegressor(length_scale, variance, noise).fit(X, y)
# same kernel, hyperparameters frozen (no optimizer), so the two should agree to numerical precision
kernel = ConstantKernel(variance, "fixed") * RBF(length_scale, "fixed")
theirs = SklearnGPR(kernel=kernel, alpha=noise**2, optimizer=None).fit(X, y)

mean_o, std_o = ours.predict(X_test, return_std=True)
mean_t, std_t = theirs.predict(X_test, return_std=True)
print("max |mean diff| = %.2e, max |std diff| = %.2e" % (np.abs(mean_o - mean_t).max(), np.abs(std_o - std_t).max()))
print("std inside data range ~%.2f, outside (x=7) ~%.2f" % (std_o[np.argmin(np.abs(X_test[:, 0] - 3))], std_o[-1]))
assert np.allclose(mean_o, mean_t, atol=1e-6)
assert np.allclose(std_o, std_t, atol=1e-5)

try:
    import matplotlib.pyplot as plt

    plt.scatter(X, y, c="k", s=15, label="data")
    plt.plot(X_test, mean_o, color="red", label="mean")
    plt.fill_between(X_test[:, 0], mean_o - 2 * std_o, mean_o + 2 * std_o, alpha=0.2, color="red", label="±2 std")
    plt.legend()
    plt.title("Gaussian Process Regression (from scratch)")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
