# From-ScratchML

Popular ML algorithms implemented in raw Python + numpy — no scikit-learn/torch in the core implementations. Each algorithm is self-contained: implementation, test, and a runnable demo that checks results against scikit-learn for correctness.

## Algorithms

| Algorithm | Type | Path |
|---|---|---|
| Linear Regression | Supervised, regression | [`algorithms/01_linear_regression`](algorithms/01_linear_regression) |
| Ridge & Lasso | Supervised, regularized regression | [`algorithms/02_ridge_lasso`](algorithms/02_ridge_lasso) |
| Elastic Net | Supervised, regularized regression | [`algorithms/03_elastic_net`](algorithms/03_elastic_net) |
| Polynomial Regression | Supervised, regression | [`algorithms/04_polynomial_regression`](algorithms/04_polynomial_regression) |
| Support Vector Regression | Supervised, regression (linear) | [`algorithms/05_support_vector_regression`](algorithms/05_support_vector_regression) |
| Gaussian Process Regression | Supervised, regression (probabilistic) | [`algorithms/06_gaussian_process_regression`](algorithms/06_gaussian_process_regression) |
| Logistic Regression | Supervised, classification | [`algorithms/07_logistic_regression`](algorithms/07_logistic_regression) |
| K-Nearest Neighbors | Supervised, classification | [`algorithms/08_knn`](algorithms/08_knn) |
| Naive Bayes | Supervised, classification | [`algorithms/09_naive_bayes`](algorithms/09_naive_bayes) |
| Decision Tree | Supervised, classification | [`algorithms/10_decision_tree`](algorithms/10_decision_tree) |
| Support Vector Machine | Supervised, classification (linear) | [`algorithms/11_svm`](algorithms/11_svm) |
| Random Forest | Supervised, classification (ensemble) | [`algorithms/12_random_forest`](algorithms/12_random_forest) |
| Gradient Boosting | Supervised, regression (ensemble) | [`algorithms/13_gradient_boosting`](algorithms/13_gradient_boosting) |
| AdaBoost | Supervised, classification (ensemble) | [`algorithms/14_adaboost`](algorithms/14_adaboost) |
| Neural Network (MLP) | Supervised, classification (deep learning) | [`algorithms/15_neural_network`](algorithms/15_neural_network) |
| K-Means | Unsupervised, clustering | [`algorithms/16_kmeans`](algorithms/16_kmeans) |
| DBSCAN | Unsupervised, density clustering | [`algorithms/17_dbscan`](algorithms/17_dbscan) |
| PCA | Unsupervised, dimensionality reduction | [`algorithms/18_pca`](algorithms/18_pca) |

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
