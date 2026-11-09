import numpy as np
from model import DBSCAN
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X, _ = make_moons(n_samples=400, noise=0.07, random_state=42)

ours = DBSCAN(eps=0.15, min_samples=5).fit(X)
theirs = SklearnDBSCAN(eps=0.15, min_samples=5).fit(X)

ari = adjusted_rand_score(ours.labels_, theirs.labels_)
print("clusters: ours=%d sklearn=%d" % (len(set(ours.labels_) - {-1}), len(set(theirs.labels_) - {-1})))
print("noise points: ours=%d sklearn=%d" % ((ours.labels_ == -1).sum(), (theirs.labels_ == -1).sum()))
print("agreement with sklearn (adjusted rand index): %.4f" % ari)
assert np.array_equal(ours.core_sample_indices_, theirs.core_sample_indices_)
assert ari > 0.99

try:
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=ours.labels_, cmap="viridis", s=15)
    plt.title("DBSCAN (from scratch): noise = purple/-1")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
