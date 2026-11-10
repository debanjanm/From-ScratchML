import numpy as np
from model import TruncatedSVD
from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD as SklearnSVD

X = load_digits().data

ours = TruncatedSVD(n_components=10).fit(X)
theirs = SklearnSVD(n_components=10, algorithm="arpack", random_state=0).fit(X)

print("singular values (top 3): ours=%s sklearn=%s" % (ours.singular_values_[:3].round(3), theirs.singular_values_[:3].round(3)))
assert np.allclose(ours.singular_values_, theirs.singular_values_, atol=1e-6)
assert np.allclose(ours.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-6)
assert np.allclose(np.abs(ours.transform(X)), np.abs(theirs.transform(X)), atol=1e-6)  # sign is arbitrary
print("projections match sklearn up to sign")

print("rank-k reconstruction error of the 64-pixel digits:")
for k in (2, 5, 10, 20, 40):
    model = TruncatedSVD(n_components=k).fit(X)
    error = np.linalg.norm(X - model.inverse_transform(model.transform(X))) / np.linalg.norm(X)
    print("  k=%2d  relative error %.3f" % (k, error))
