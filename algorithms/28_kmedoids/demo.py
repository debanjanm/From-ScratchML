import numpy as np
from model import KMedoids
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# clean data: K-Medoids and K-Means should find the same clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
medoids = KMedoids(n_clusters=3).fit(X)
means = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
ari = adjusted_rand_score(medoids.labels_, means.labels_)
print("clean blobs: agreement with K-Means (ARI) = %.4f" % ari)
assert ari > 0.95

# two blobs centred at (0,0) and (20,0), plus 3 extreme outliers far to the left of the first
rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal([0, 0], 0.5, (60, 2)),
    rng.normal([20, 0], 0.5, (60, 2)),
    [[-30.0, 0.0], [-31.0, 1.0], [-29.0, -1.0]],
])
medoids = KMedoids(n_clusters=2).fit(X)
means = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

def error_at_first_blob(centers):  # distance from the true centre (0, 0) to the nearest found centre
    return np.linalg.norm(centers - np.array([0.0, 0.0]), axis=1).min()

e_medoid = error_at_first_blob(medoids.cluster_centers_)
e_mean = error_at_first_blob(means.cluster_centers_)
print("with outliers, centre of the first blob is off by: medoid=%.2f  kmeans=%.2f" % (e_medoid, e_mean))
assert e_medoid < e_mean
