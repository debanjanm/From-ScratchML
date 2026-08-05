# From-ScratchML

Popular ML algorithms implemented in raw Python + numpy — no scikit-learn/torch in the core implementations. Each algorithm is self-contained: implementation, test, and a runnable demo that checks results against scikit-learn for correctness.

## Algorithms

| Algorithm | Type | Path |
|---|---|---|
| Linear Regression | Supervised, regression | [`algorithms/linear_regression`](algorithms/linear_regression) |
| Logistic Regression | Supervised, classification | [`algorithms/logistic_regression`](algorithms/logistic_regression) |
| K-Nearest Neighbors | Supervised, classification | [`algorithms/knn`](algorithms/knn) |
| Decision Tree | Supervised, classification | [`algorithms/decision_tree`](algorithms/decision_tree) |
| Naive Bayes | Supervised, classification | [`algorithms/naive_bayes`](algorithms/naive_bayes) |
| K-Means | Unsupervised, clustering | [`algorithms/kmeans`](algorithms/kmeans) |
| PCA | Unsupervised, dimensionality reduction | [`algorithms/pca`](algorithms/pca) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Each algorithm folder has the same layout:

```
algorithms/<name>/
├── model.py       # the from-scratch implementation
├── test_model.py  # assert-based correctness checks
├── demo.py        # runnable example, compares against scikit-learn
└── README.md      # math + usage notes
```

Run any algorithm's test or demo directly:

```bash
python algorithms/linear_regression/test_model.py
python algorithms/linear_regression/demo.py
```
