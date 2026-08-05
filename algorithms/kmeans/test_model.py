import numpy as np
from model import KMeans


def test_three_separated_blobs():
    rng = np.random.default_rng(0)
    blob0 = rng.normal(loc=[0, 0], scale=0.3, size=(50, 2))
    blob1 = rng.normal(loc=[10, 10], scale=0.3, size=(50, 2))
    blob2 = rng.normal(loc=[0, 10], scale=0.3, size=(50, 2))
    X = np.vstack([blob0, blob1, blob2])

    model = KMeans(n_clusters=3, random_state=1).fit(X)

    # each true blob should map to exactly one cluster label (up to permutation)
    labels_per_blob = [
        set(model.labels_[0:50]),
        set(model.labels_[50:100]),
        set(model.labels_[100:150]),
    ]
    for labels in labels_per_blob:
        assert len(labels) == 1
    assert len({frozenset(s) for s in labels_per_blob}) == 3  # 3 distinct clusters used


def test_inertia_decreases_with_more_clusters():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    inertia_1 = KMeans(n_clusters=1, random_state=0).fit(X).inertia_
    inertia_5 = KMeans(n_clusters=5, random_state=0).fit(X).inertia_
    assert inertia_5 < inertia_1


if __name__ == "__main__":
    test_three_separated_blobs()
    test_inertia_decreases_with_more_clusters()
    print("all tests passed")
