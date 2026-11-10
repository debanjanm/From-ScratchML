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
| Decision Tree Regressor | Supervised, regression (tree) | [`algorithms/07_decision_tree_regressor`](algorithms/07_decision_tree_regressor) |
| Perceptron | Supervised, classification (linear) | [`algorithms/08_perceptron`](algorithms/08_perceptron) |
| Logistic Regression | Supervised, classification | [`algorithms/09_logistic_regression`](algorithms/09_logistic_regression) |
| Softmax Regression | Supervised, multiclass classification | [`algorithms/10_softmax_regression`](algorithms/10_softmax_regression) |
| K-Nearest Neighbors | Supervised, classification | [`algorithms/11_knn`](algorithms/11_knn) |
| Gaussian Naive Bayes | Supervised, classification (probabilistic) | [`algorithms/12_naive_bayes`](algorithms/12_naive_bayes) |
| Multinomial & Bernoulli Naive Bayes | Supervised, classification (probabilistic) | [`algorithms/13_multinomial_bernoulli_nb`](algorithms/13_multinomial_bernoulli_nb) |
| LDA & QDA | Supervised, classification (probabilistic) | [`algorithms/14_discriminant_analysis`](algorithms/14_discriminant_analysis) |
| Decision Tree | Supervised, classification (tree) | [`algorithms/15_decision_tree`](algorithms/15_decision_tree) |
| Support Vector Machine | Supervised, classification (linear) | [`algorithms/16_svm`](algorithms/16_svm) |
| Kernel SVM (SMO) | Supervised, classification (non-linear) | [`algorithms/17_kernel_svm`](algorithms/17_kernel_svm) |
| Random Forest | Supervised, classification (ensemble) | [`algorithms/18_random_forest`](algorithms/18_random_forest) |
| Gradient Boosting | Supervised, regression (ensemble) | [`algorithms/19_gradient_boosting`](algorithms/19_gradient_boosting) |
| AdaBoost | Supervised, classification (ensemble) | [`algorithms/20_adaboost`](algorithms/20_adaboost) |
| Neural Network (MLP) | Supervised, classification (deep learning) | [`algorithms/21_neural_network`](algorithms/21_neural_network) |
| K-Means | Unsupervised, clustering | [`algorithms/22_kmeans`](algorithms/22_kmeans) |
| DBSCAN | Unsupervised, density clustering | [`algorithms/23_dbscan`](algorithms/23_dbscan) |
| PCA | Unsupervised, dimensionality reduction | [`algorithms/24_pca`](algorithms/24_pca) |

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
