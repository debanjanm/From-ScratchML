import numpy as np
from model import AgglomerativeClustering


def test_merge_heights_match_scipy_for_every_linkage():
    from scipy.cluster.hierarchy import linkage

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    for method in ("single", "complete", "average", "ward"):
        ours = AgglomerativeClustering(n_clusters=2, linkage=method).fit(X).linkage_matrix_
        theirs = linkage(X, method=method)
        assert np.allclose(np.sort(ours[:, 2]), np.sort(theirs[:, 2])), method
        assert ours[-1, 3] == 60  # the last merge contains every point


def test_merge_heights_never_decrease_for_monotone_linkages():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    for method in ("single", "complete", "average", "ward"):
        heights = AgglomerativeClustering(linkage=method).fit(X).linkage_matrix_[:, 2]
        assert (np.diff(heights) >= -1e-9).all(), method


def test_cutting_the_tree_recovers_separated_blobs():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(c, 0.4, (30, 2)) for c in ([0, 0], [8, 0], [0, 8])])
    for method in ("single", "complete", "average", "ward"):
        labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X)
        assert len(set(labels[:30])) == len(set(labels[30:60])) == len(set(labels[60:])) == 1, method
        assert len(set(labels)) == 3, method


def test_single_linkage_follows_a_chain_but_complete_does_not():
    # two long thin parallel lines: single linkage chains along each, complete linkage cuts across
    line = np.linspace(0, 10, 30)
    X = np.vstack([np.column_stack([line, np.zeros(30)]), np.column_stack([line, np.full(30, 3.0)])])
    single = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X)
    complete = AgglomerativeClustering(n_clusters=2, linkage="complete").fit_predict(X)
    assert len(set(single[:30])) == 1 and len(set(single[30:])) == 1 and single[0] != single[-1]
    assert len(set(complete[:30])) > 1 or len(set(complete[30:])) > 1


if __name__ == "__main__":
    test_merge_heights_match_scipy_for_every_linkage()
    test_merge_heights_never_decrease_for_monotone_linkages()
    test_cutting_the_tree_recovers_separated_blobs()
    test_single_linkage_follows_a_chain_but_complete_does_not()
    print("all tests passed")
