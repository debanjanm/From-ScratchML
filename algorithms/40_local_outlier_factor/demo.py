import numpy as np
from model import LocalOutlierFactor
from sklearn.neighbors import LocalOutlierFactor as SklearnLOF

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 0.1, size=(150, 2)), rng.normal([20, 0], 2.0, size=(150, 2)), [[1.0, 0.0], [20.0, 12.0]]])
X_new = np.array([[0.0, 0.0], [20.0, 0.0], [1.0, 0.0], [10.0, 5.0]])

ours = LocalOutlierFactor(n_neighbors=15, contamination=0.02).fit(X)
theirs = SklearnLOF(n_neighbors=15, contamination=0.02).fit(X)
assert np.allclose(ours.negative_outlier_factor_, theirs.negative_outlier_factor_)
assert (ours.fit_predict(X) == theirs.fit_predict(X)).all()
print("LOF scores for all %d training points identical to sklearn" % len(X))

novel = SklearnLOF(n_neighbors=15, contamination=0.02, novelty=True).fit(X)
assert np.allclose(ours.score_samples(X_new), novel.score_samples(X_new))
assert (ours.predict(X_new) == novel.predict(X_new)).all()
print("scores for new points identical to sklearn (novelty=True): %s" % (-ours.score_samples(X_new)).round(2))

lof = -ours.negative_outlier_factor_
top = np.argsort(lof)[::-1][:2]
print("top-2 LOF: points %s with LOF %s (the two planted ones are %d and %d)" % (top, lof[top].round(2), len(X) - 2, len(X) - 1))
assert set(top) == {len(X) - 2, len(X) - 1}

# a global rule ("distance to the 15th neighbour") would rank the sparse cluster's members above the point at (1, 0)
print("(1,0) ranks #%d by k-distance, but #%d by LOF" % ((ours.k_distance_ > ours.k_distance_[-2]).sum() + 1, (lof > lof[-2]).sum() + 1))
