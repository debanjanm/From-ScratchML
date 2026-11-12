import numpy as np
from model import IsolationForest
from scipy.stats import spearmanr
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest as SklearnIF
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
X_in, _ = make_blobs(n_samples=950, centers=[[0, 0], [6, 6]], cluster_std=1.0, random_state=0)
X_out = rng.uniform(-6, 12, size=(50, 2))
X = np.vstack([X_in, X_out])
is_outlier = np.array([0] * 950 + [1] * 50)   # note: some "outliers" land inside the blobs by chance

ours = IsolationForest(n_estimators=100, contamination=0.05, random_state=0).fit(X)
theirs = SklearnIF(n_estimators=100, contamination=0.05, random_state=0).fit(X)

s_ours = ours.score_samples(X)
s_theirs = -theirs.score_samples(X)   # sklearn returns the negative score
rho = spearmanr(s_ours, s_theirs)[0]
print("ROC AUC at spotting the planted outliers: ours=%.3f sklearn=%.3f" % (roc_auc_score(is_outlier, s_ours), roc_auc_score(is_outlier, s_theirs)))
print("rank correlation of the two scores (different random trees, so not identical): %.3f" % rho)
print("flagged: ours=%d sklearn=%d, agree on %.1f%% of labels" % ((ours.predict(X) == -1).sum(), (theirs.predict(X) == -1).sum(), 100 * (ours.predict(X) == theirs.predict(X)).mean()))
assert rho > 0.9
assert abs(roc_auc_score(is_outlier, s_ours) - roc_auc_score(is_outlier, s_theirs)) < 0.02
