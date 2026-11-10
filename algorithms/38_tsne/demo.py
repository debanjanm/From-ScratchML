import numpy as np
from model import TSNE
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as SklearnTSNE, trustworthiness

digits = load_digits()
X, y = digits.data[:600], digits.target[:600]

ours = TSNE(perplexity=30, n_iter=1000, random_state=0)
Y_ours = ours.fit_transform(X)
sk = SklearnTSNE(perplexity=30, max_iter=1000, init="random", random_state=0, method="exact")
Y_theirs = sk.fit_transform(X)
Y_pca = PCA(2).fit_transform(X)

nearest = lambda Y: (y[np.argsort(((Y[:, None] - Y[None]) ** 2).sum(-1), axis=1)[:, 1]] == y).mean()
print("%-8s %-10s %-24s %s" % ("", "KL", "trustworthiness (k=10)", "1-NN label purity in 2-D"))
print("%-8s %-10.4f %-24.4f %.3f" % ("ours", ours.kl_divergence_, trustworthiness(X, Y_ours, n_neighbors=10), nearest(Y_ours)))
print("%-8s %-10.4f %-24.4f %.3f" % ("sklearn", sk.kl_divergence_, trustworthiness(X, Y_theirs, n_neighbors=10), nearest(Y_theirs)))
print("%-8s %-10s %-24.4f %.3f" % ("PCA", "-", trustworthiness(X, Y_pca, n_neighbors=10), nearest(Y_pca)))

# different random starts -> different pictures, but the quality should be comparable
assert abs(ours.kl_divergence_ - sk.kl_divergence_) < 0.15
assert trustworthiness(X, Y_ours, n_neighbors=10) > 0.97
assert nearest(Y_ours) > 0.9 and nearest(Y_ours) > nearest(Y_pca) + 0.3

try:
    import matplotlib.pyplot as plt

    plt.scatter(Y_ours[:, 0], Y_ours[:, 1], c=y, cmap="tab10", s=8)
    plt.title("t-SNE on 600 digits (from scratch)")
    plt.savefig("demo_plot.png")
    print("saved plot to demo_plot.png")
except Exception as e:
    print(f"skipping plot: {e}")
