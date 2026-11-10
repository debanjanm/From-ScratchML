import numpy as np
from model import MeanShift, estimate_bandwidth
from sklearn.cluster import MeanShift as SklearnMeanShift
from sklearn.cluster import estimate_bandwidth as sklearn_estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, truth = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=11)

bw_ours, bw_theirs = estimate_bandwidth(X, 0.2), sklearn_estimate_bandwidth(X, quantile=0.2)
print("estimated bandwidth: ours=%.4f sklearn=%.4f" % (bw_ours, bw_theirs))
assert np.isclose(bw_ours, bw_theirs)

ours = MeanShift(bandwidth=bw_ours).fit(X)
theirs = SklearnMeanShift(bandwidth=bw_theirs).fit(X)
print("clusters found (k was never given): ours=%d sklearn=%d" % (len(ours.cluster_centers_), len(theirs.cluster_centers_)))
print("modes agree: max centre difference after sorting = %.2e"
      % np.abs(np.sort(ours.cluster_centers_, axis=0) - np.sort(theirs.cluster_centers_, axis=0)).max())
print("agreement with sklearn labels (ARI) = %.4f   with true blobs = %.4f"
      % (adjusted_rand_score(ours.labels_, theirs.labels_), adjusted_rand_score(ours.labels_, truth)))
assert len(ours.cluster_centers_) == len(theirs.cluster_centers_)
assert adjusted_rand_score(ours.labels_, theirs.labels_) > 0.99
