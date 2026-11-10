import numpy as np
from model import LDAProjection
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

data = load_wine()
X, y = data.data, data.target

ours = LDAProjection().fit(X, y)
theirs = LinearDiscriminantAnalysis(solver="eigen", n_components=2).fit(X, y)

print("explained ratio: ours=%s sklearn=%s" % (ours.explained_variance_ratio_.round(4), theirs.explained_variance_ratio_.round(4)))
assert np.allclose(ours.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-6)

# scaling conventions differ, so compare each discriminant axis by correlation (sign-free)
a, b = ours.transform(X), theirs.transform(X)
for j in range(2):
    corr = abs(np.corrcoef(a[:, j], b[:, j])[0, 1])
    print("discriminant %d: |correlation with sklearn| = %.6f" % (j + 1, corr))
    assert corr > 0.9999

# why supervised matters: 2-D wine, 5-fold kNN accuracy
Xs = (X - X.mean(0)) / X.std(0)
knn = KNeighborsClassifier(5)
acc_lda = cross_val_score(knn, LDAProjection().fit_transform(Xs, y), y, cv=5).mean()
acc_pca = cross_val_score(knn, PCA(2).fit_transform(Xs), y, cv=5).mean()
print("kNN accuracy on 2-D embedding: LDA=%.3f  PCA=%.3f" % (acc_lda, acc_pca))
