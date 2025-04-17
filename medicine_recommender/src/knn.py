import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

class KnnClustering():
    def __init__(self, n_neighbors, df):
        self.n_neighbors = n_neighbors
        self.df = df
        self.knn = None
        self.encoders = [MultiLabelBinarizer(), MultiLabelBinarizer(), MultiLabelBinarizer(), LabelEncoder()]

    def fit_encoders(self, X, y):
        """
        This function fits the data encoders to the training data.
        """
        self.encoders[0].fit(X['Symptoms'])
        self.encoders[1].fit(X['Disease'])
        self.encoders[2].fit(X['Causes'])
        self.encoders[3].fit(y)

    def encode_features(self, X):
        """
        This prepares a feature matrix for the KNN model.
        """
        X_symptoms = self.encoders[0].transform(X['Symptoms'])
        X_diseases = self.encoders[1].transform(X['Disease'])
        X_causes = self.encoders[2].transform(X['Causes'])
        X_transformed = np.hstack((X_symptoms, X_diseases, X_causes))
        return X_transformed       

    def train(self):
        """
        This function trains the KNN model.
        """
        X = self.df[['Symptoms', 'Disease', 'Causes']]
        y = self.df['Medicine'].apply(lambda x: x[0])
        self.fit_encoders(X, y)
        X_transformed = self.encode_features(X)
        y_transformed = self.encoders[3].transform(y)
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='cosine')
        self.knn.fit(X_transformed, y_transformed)

    def infer(self, symptoms, disease, causes):
        """
        This function returns medicine predictions for the KNN model,
        from new user input data
        """
        df_infer = pd.DataFrame({'Symptoms': [symptoms], 
                        'Disease': [disease], 
                        'Causes': [causes]})
        X_transformed = self.encode_features(df_infer)
        predicted_ids = self.knn.predict(X_transformed)
        predicted_medicines = self.encoders[3].inverse_transform(predicted_ids)
        return predicted_medicines[0]