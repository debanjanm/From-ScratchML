import numpy as np
from model import XGBoost
from sklearn.datasets import load_breast_cancer, make_friedman1
from sklearn.model_selection import train_test_split

PARAMS = dict(n_estimators=100, learning_rate=0.3, max_depth=3)

# --- regression ---
X, y = make_friedman1(n_samples=500, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
ours = XGBoost("reg:squarederror", **PARAMS).fit(X_train, y_train)
mse_ours = np.mean((ours.predict(X_test) - y_test) ** 2)

try:
    import xgboost as xgb  # optional: pip install xgboost to compare against the real library

    theirs = xgb.XGBRegressor(**PARAMS, tree_method="exact", base_score=float(y_train.mean())).fit(X_train, y_train)
    mse_theirs = np.mean((theirs.predict(X_test) - y_test) ** 2)
    print("regression test MSE: ours=%.4f xgboost=%.4f" % (mse_ours, mse_theirs))
    assert abs(mse_ours - mse_theirs) / mse_theirs < 0.05
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    theirs = GradientBoostingRegressor(n_estimators=100, learning_rate=0.3, max_depth=3, random_state=0).fit(X_train, y_train)
    mse_theirs = np.mean((theirs.predict(X_test) - y_test) ** 2)
    print("regression test MSE: ours=%.4f sklearn GBR=%.4f  (xgboost not installed)" % (mse_ours, mse_theirs))
    assert abs(mse_ours - mse_theirs) / mse_theirs < 0.15

# --- binary classification ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
ours = XGBoost("binary:logistic", **PARAMS).fit(X_train, y_train)
acc_ours = (ours.predict(X_test) == y_test).mean()
try:
    p0 = float(y_train.mean())
    theirs = xgb.XGBClassifier(**PARAMS, tree_method="exact", base_score=p0).fit(X_train, y_train)
    acc_theirs = (theirs.predict(X_test) == y_test).mean()
    print("breast cancer accuracy: ours=%.4f xgboost=%.4f" % (acc_ours, acc_theirs))
    assert abs(acc_ours - acc_theirs) < 0.03
except NameError:
    print("breast cancer accuracy: ours=%.4f" % acc_ours)
    assert acc_ours > 0.93
