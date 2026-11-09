# From-ScratchML

Popular ML algorithms implemented in raw Python + numpy — no scikit-learn/torch in the core implementations. Each algorithm is self-contained: implementation, test, and a runnable demo that checks results against scikit-learn for correctness.

## Algorithms

| Algorithm | Type | Path |
|---|---|---|
| Linear Regression | Supervised, regression | [`algorithms/01_linear_regression`](algorithms/01_linear_regression) |
| Ridge & Lasso | Supervised, regularized regression | [`algorithms/02_ridge_lasso`](algorithms/02_ridge_lasso) |
| Logistic Regression | Supervised, classification | [`algorithms/03_logistic_regression`](algorithms/03_logistic_regression) |
| K-Nearest Neighbors | Supervised, classification | [`algorithms/04_knn`](algorithms/04_knn) |
| Naive Bayes | Supervised, classification | [`algorithms/05_naive_bayes`](algorithms/05_naive_bayes) |
| Decision Tree | Supervised, classification | [`algorithms/06_decision_tree`](algorithms/06_decision_tree) |
| Support Vector Machine | Supervised, classification (linear) | [`algorithms/07_svm`](algorithms/07_svm) |
| Random Forest | Supervised, classification (ensemble) | [`algorithms/08_random_forest`](algorithms/08_random_forest) |
| Gradient Boosting | Supervised, regression (ensemble) | [`algorithms/09_gradient_boosting`](algorithms/09_gradient_boosting) |
| AdaBoost | Supervised, classification (ensemble) | [`algorithms/10_adaboost`](algorithms/10_adaboost) |
| Neural Network (MLP) | Supervised, classification (deep learning) | [`algorithms/11_neural_network`](algorithms/11_neural_network) |
| K-Means | Unsupervised, clustering | [`algorithms/12_kmeans`](algorithms/12_kmeans) |
| DBSCAN | Unsupervised, density clustering | [`algorithms/13_dbscan`](algorithms/13_dbscan) |
| PCA | Unsupervised, dimensionality reduction | [`algorithms/14_pca`](algorithms/14_pca) |

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
