import numpy as np
from model import KMeans
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

ours = KMeans(n_clusters=4, random_state=0).fit(X)
theirs = SklearnKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)

print("inertia: ours=%.4f sklearn=%.4f" % (ours.inertia_, theirs.inertia_))
# k-means is sensitive to init; a single-init run should still land close to sklearn's n_init=10 optimum
assert ours.inertia_ < theirs.inertia_ * 1.5

try:
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=ours.labels_, cmap="viridis", s=15, alpha=0.6)
    plt.scatter(ours.centroids[:, 0], ours.centroids[:, 1], c="red", marker="x", s=100)
    plt.title("K-Means (from scratch)")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
