import numpy as np
from model import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SklearnGMM

X, truth = make_blobs(n_samples=600, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=3)

ours = GaussianMixture(n_components=3, tol=1e-10, max_iter=1000).fit(X)
theirs = SklearnGMM(n_components=3, covariance_type="full", tol=1e-10, max_iter=1000, reg_covar=1e-6, random_state=0).fit(X)

print("avg log-likelihood: ours=%.6f sklearn=%.6f" % (ours.score(X), theirs.score(X)))
print("agreement with sklearn labels (ARI) = %.4f" % adjusted_rand_score(ours.predict(X), theirs.predict(X)))
print("agreement with true blobs (ARI)     = %.4f" % adjusted_rand_score(ours.predict(X), truth))
print("EM iterations: %d, log-likelihood %.3f -> %.3f"
      % (len(ours.log_likelihood_history_), ours.log_likelihood_history_[0], ours.log_likelihood_history_[-1]))
assert abs(ours.score(X) - theirs.score(X)) < 1e-4
assert np.allclose(np.sort(ours.means_, axis=0), np.sort(theirs.means_, axis=0), atol=1e-3)
assert adjusted_rand_score(ours.predict(X), theirs.predict(X)) > 0.99
