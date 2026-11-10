import numpy as np
from model import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

X, truth = make_blobs(n_samples=200, centers=3, cluster_std=0.9, random_state=7)
print("blobs: merge heights vs scipy, labels vs sklearn")
for method in ("single", "complete", "average", "ward"):
    ours = AgglomerativeClustering(n_clusters=3, linkage=method).fit(X)
    heights_match = np.allclose(np.sort(ours.linkage_matrix_[:, 2]), np.sort(linkage(X, method)[:, 2]))
    ari = adjusted_rand_score(ours.labels_, SklearnAgglomerative(n_clusters=3, linkage=method).fit_predict(X))
    print("  %-9s heights match scipy: %s   label agreement with sklearn (ARI): %.4f" % (method, heights_match, ari))
    assert heights_match and ari > 0.99

# the linkage choice changes WHAT is found: on two moons only single linkage follows the curves
Xm, truth_m = make_moons(n_samples=200, noise=0.05, random_state=0)
print("moons, agreement with the true shapes (ARI):")
for method in ("single", "ward"):
    labels = AgglomerativeClustering(n_clusters=2, linkage=method).fit_predict(Xm)
    print("  %-9s %.4f" % (method, adjusted_rand_score(truth_m, labels)))
assert adjusted_rand_score(truth_m, AgglomerativeClustering(2, "single").fit_predict(Xm)) > 0.95
assert adjusted_rand_score(truth_m, AgglomerativeClustering(2, "ward").fit_predict(Xm)) < 0.6
