import numpy as np
from model import PCA
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

data = load_iris()
X = data.data

ours = PCA(n_components=2).fit(X)
theirs = SklearnPCA(n_components=2).fit(X)

print("explained variance ratio: ours=%s sklearn=%s" % (ours.explained_variance_ratio_, theirs.explained_variance_ratio_))
assert np.allclose(ours.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-6)

ours_proj = ours.transform(X)
theirs_proj = theirs.transform(X)
# eigenvector sign is arbitrary, so compare magnitude/shape rather than raw sign-sensitive values
assert np.allclose(np.abs(ours_proj), np.abs(theirs_proj), atol=1e-4)
print("projections match sklearn up to sign")

try:
    import matplotlib.pyplot as plt

    plt.scatter(ours_proj[:, 0], ours_proj[:, 1], c=data.target, cmap="viridis", s=15)
    plt.title("PCA on Iris (from scratch)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
