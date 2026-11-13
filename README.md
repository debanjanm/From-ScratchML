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
| Bagging | Ensemble (parallel) | [`algorithms/18_bagging`](algorithms/18_bagging) |
| Random Forest | Ensemble (parallel) | [`algorithms/19_random_forest`](algorithms/19_random_forest) |
| Extra Trees | Ensemble (parallel) | [`algorithms/20_extra_trees`](algorithms/20_extra_trees) |
| AdaBoost | Ensemble (boosting) | [`algorithms/21_adaboost`](algorithms/21_adaboost) |
| Gradient Boosting | Ensemble (boosting), regression | [`algorithms/22_gradient_boosting`](algorithms/22_gradient_boosting) |
| Gradient Boosting Classifier | Ensemble (boosting), classification | [`algorithms/23_gradient_boosting_classifier`](algorithms/23_gradient_boosting_classifier) |
| XGBoost | Ensemble (boosting), second-order | [`algorithms/24_xgboost`](algorithms/24_xgboost) |
| Voting & Stacking | Ensemble (combining different models) | [`algorithms/25_voting_stacking`](algorithms/25_voting_stacking) |
| Neural Network (MLP) | Supervised, classification (deep learning) | [`algorithms/26_neural_network`](algorithms/26_neural_network) |
| K-Means | Unsupervised, clustering (centroid) | [`algorithms/27_kmeans`](algorithms/27_kmeans) |
| K-Medoids | Unsupervised, clustering (medoid, any metric) | [`algorithms/28_kmedoids`](algorithms/28_kmedoids) |
| Gaussian Mixture (EM) | Unsupervised, clustering (probabilistic) | [`algorithms/29_gaussian_mixture`](algorithms/29_gaussian_mixture) |
| Hierarchical Clustering | Unsupervised, clustering (tree) | [`algorithms/30_hierarchical_clustering`](algorithms/30_hierarchical_clustering) |
| Mean Shift | Unsupervised, clustering (mode seeking) | [`algorithms/31_mean_shift`](algorithms/31_mean_shift) |
| DBSCAN | Unsupervised, clustering (density) | [`algorithms/32_dbscan`](algorithms/32_dbscan) |
| Spectral Clustering | Unsupervised, clustering (graph) | [`algorithms/33_spectral_clustering`](algorithms/33_spectral_clustering) |
| PCA | Unsupervised, dimensionality reduction | [`algorithms/34_pca`](algorithms/34_pca) |
| Truncated SVD | Unsupervised, dimensionality reduction | [`algorithms/35_svd`](algorithms/35_svd) |
| Kernel PCA | Unsupervised, dimensionality reduction | [`algorithms/36_kernel_pca`](algorithms/36_kernel_pca) |
| LDA Projection | Supervised, dimensionality reduction | [`algorithms/37_lda_projection`](algorithms/37_lda_projection) |
| t-SNE | Unsupervised, dimensionality reduction (visualization) | [`algorithms/38_tsne`](algorithms/38_tsne) |
| Isolation Forest | Unsupervised, anomaly detection | [`algorithms/39_isolation_forest`](algorithms/39_isolation_forest) |
| Local Outlier Factor | Unsupervised, anomaly detection | [`algorithms/40_local_outlier_factor`](algorithms/40_local_outlier_factor) |
| Bayesian Linear Regression | Supervised, regression (probabilistic) | [`algorithms/41_bayesian_linear_regression`](algorithms/41_bayesian_linear_regression) |
| Hidden Markov Model | Probabilistic, sequence model | [`algorithms/42_hidden_markov_model`](algorithms/42_hidden_markov_model) |

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
