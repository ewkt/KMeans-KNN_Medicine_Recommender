import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

class KMeansClustering():
    def __init__(self, nb_clusters, df):
        """
        This class is used to train the model.
        It includes functions to load data, train the model, and save the model.
        """
        self.df = df
        self.nb_clusters = nb_clusters
        self.kmeans = KMeans(n_clusters=self.nb_clusters, random_state=113)
        self.mlb_symptoms = MultiLabelBinarizer()
        self.mlb_disease = MultiLabelBinarizer()
        self.mlb_causes = MultiLabelBinarizer()

    def fit_mlbs(self, X):
        """
        This function fits the MultiLabelBinarizer to the data.
        """
        self.mlb_symptoms.fit(X['Symptoms'])
        self.mlb_disease.fit(X['Disease'])
        self.mlb_causes.fit(X['Causes'])

    def pipeline(self, X, train=False):
        """
        This function takes a pandas DataFrame and applies the training pipeline to it.
        """
        X_symptoms = self.mlb_symptoms.transform(X['Symptoms'])
        X_diseases = self.mlb_disease.transform(X['Disease'])
        X_causes = self.mlb_causes.transform(X['Causes'])
        X_transformed = np.hstack((X_symptoms, X_diseases, X_causes))
        if(train):
            self.kmeans.fit(X_transformed)
            clusters = self.kmeans.labels_

        else:
            clusters = self.kmeans.predict(X_transformed)

        return clusters
    
    def top_medicines(self, X, clusters):
        """
        This function builds the top medicines for each cluster.
        """
        X['Cluster'] = clusters
        cluster_top_medecine = {}

        for cluster in range(self.nb_clusters):
            cluster_df = X[X['Cluster'] == cluster]
            top_medicine = cluster_df['Medicine'].explode().value_counts().head(1).index[0]
            cluster_top_medecine[cluster] = top_medicine
        
        return cluster_top_medecine

    def train(self):
        """
        This function runs the main pipeline.
        """
        X_train = self.df[['Symptoms', 'Disease', 'Causes', 'Medicine']]
        self.fit_mlbs(X_train)
        clusters = self.pipeline(X=X_train, train=True)
        self.cluster_top_medicine = self.top_medicines(X_train, clusters)

    def infer(self, symptoms, disease, causes):
        df_infer = pd.DataFrame({'Symptoms': [symptoms], 
                              'Disease': [disease], 
                              'Causes': [causes]})
        
        cluster_new = self.pipeline(X=df_infer, train=False)
        predicted_medicine = self.cluster_top_medicine[cluster_new[0]]
        return predicted_medicine