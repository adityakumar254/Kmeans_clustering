from ml_model.clustering import ClusteringModel
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Initialize clustering model
    model = ClusteringModel(n_clusters=5)
    
    # Load the model
    model.load_model("kmeans.pkl")
    
    # Load the data for prediction
    data_path = "data/Mall_Customers.csv"
    df = pd.read_csv(data_path)
    
    # Preprocess the data
    features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    train_data, test_data = train_test_split(features, test_size=0.2, random_state=42)
    
    # Make predictions
    predictions = model.predict(test_data)
    
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
