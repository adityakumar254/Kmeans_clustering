import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class ClusteringModel:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
    
    def load_data(self, path):
        """Load dataset from a CSV file."""
        df = pd.read_csv("data/Mall_Customers.csv")
        return df

    def preprocess_data(self, df):
        """Preprocess data for clustering."""
        features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features
    
    def train_model(self, data):
        """Train the KMeans model."""
        self.model.fit(data)
        return self.model.labels_
    
    def save_model(self, model_path="kmeans.pkl"):
        """Save the trained model to disk."""
        joblib.dump(self.model, model_path)

    def load_model(self, model_path="kmeans.pkl"):
        """Load a previously saved model."""
        self.model = joblib.load(model_path)
    
    def predict(self, data):
        """Predict using the trained model."""
        return self.model.predict(data)

