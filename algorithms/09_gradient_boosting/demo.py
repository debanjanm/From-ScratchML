from model import GradientBoostingRegressor
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=500, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ours = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
theirs = SklearnGBR(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_train, y_train)

mse_ours = mean_squared_error(y_test, ours.predict(X_test))
mse_theirs = mean_squared_error(y_test, theirs.predict(X_test))
print("friedman1 test MSE: ours=%.4f sklearn=%.4f" % (mse_ours, mse_theirs))
assert abs(mse_ours - mse_theirs) / mse_theirs < 0.15
