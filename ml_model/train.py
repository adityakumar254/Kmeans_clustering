from ml_model.clustering import ClusteringModel

def main():
    # Initialize clustering model with 5 clusters
    model = ClusteringModel(n_clusters=5)
    
    # Load and preprocess the data
    data_path = "data/Mall_Customers.csv"
    df = model.load_data(data_path)
    processed_data = model.preprocess_data(df)
    
    # Train the model
    clusters = model.train_model(processed_data)
    
    # Save the trained model
    model.save_model("kmeans.pkl")
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()

