# KMeans_Medicine_Recommender
App for Predicting Medicine from a synthetic Kaggle Dataset

Dataset used: [here](https://www.kaggle.com/datasets/joymarhew/medical-reccomadation-dataset). Note: this dataset is very small and entirely synthetic with a lot of un-clean data.

Features: One-hot encodided 'Diseases', 'Causes' and 'Symptoms'

Models used: scikit-learn's [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

Performance: **90% Accuracy** on a random test set (50 occurences) with 15 clusters. (see notebook)
