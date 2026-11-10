from model import SpectralClustering
from sklearn.cluster import KMeans, SpectralClustering as SklearnSpectral
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

cases = {
    "two moons": (make_moons(n_samples=300, noise=0.06, random_state=1), dict(affinity="nearest_neighbors", n_neighbors=10)),
    "concentric circles": (make_circles(n_samples=300, factor=0.4, noise=0.05, random_state=1), dict(affinity="rbf", gamma=20.0)),
}

for name, ((X, truth), params) in cases.items():
    ours = SpectralClustering(n_clusters=2, **params).fit_predict(X)
    theirs = SklearnSpectral(n_clusters=2, random_state=0, **params).fit_predict(X)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    ari = lambda labels: adjusted_rand_score(truth, labels)
    print("%-19s agreement with the true shapes (ARI): ours=%.3f  sklearn=%.3f  kmeans=%.3f"
          % (name, ari(ours), ari(theirs), ari(kmeans)))
    assert ari(ours) > 0.95 and ari(theirs) > 0.95
    assert ari(kmeans) < 0.6
