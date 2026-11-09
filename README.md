# From-ScratchML

Popular ML algorithms implemented in raw Python + numpy — no scikit-learn/torch in the core implementations. Each algorithm is self-contained: implementation, test, and a runnable demo that checks results against scikit-learn for correctness.

## Algorithms

| Algorithm | Type | Path |
|---|---|---|
| Linear Regression | Supervised, regression | [`algorithms/01_linear_regression`](algorithms/01_linear_regression) |
| Logistic Regression | Supervised, classification | [`algorithms/02_logistic_regression`](algorithms/02_logistic_regression) |
| K-Nearest Neighbors | Supervised, classification | [`algorithms/03_knn`](algorithms/03_knn) |
| Decision Tree | Supervised, classification | [`algorithms/04_decision_tree`](algorithms/04_decision_tree) |
| Naive Bayes | Supervised, classification | [`algorithms/05_naive_bayes`](algorithms/05_naive_bayes) |
| K-Means | Unsupervised, clustering | [`algorithms/06_kmeans`](algorithms/06_kmeans) |
| PCA | Unsupervised, dimensionality reduction | [`algorithms/07_pca`](algorithms/07_pca) |
| Random Forest | Supervised, classification (ensemble) | [`algorithms/08_random_forest`](algorithms/08_random_forest) |
| Gradient Boosting | Supervised, regression (ensemble) | [`algorithms/09_gradient_boosting`](algorithms/09_gradient_boosting) |
| Support Vector Machine | Supervised, classification (linear) | [`algorithms/10_svm`](algorithms/10_svm) |
| Neural Network (MLP) | Supervised, classification (deep learning) | [`algorithms/11_neural_network`](algorithms/11_neural_network) |

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
python algorithms/01_linear_regression/test_model.py
python algorithms/01_linear_regression/demo.py
```
