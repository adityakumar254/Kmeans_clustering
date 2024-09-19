from ml_model.clustering import ClusteringModel
import numpy as np
import pytest

def test_clustering_model():
    model = ClusteringModel(n_clusters=5)
    
    # Create some sample data
    sample_data = np.array([[15, 39], [16, 81], [17, 6], [18, 77], [19, 40]])
    
    # Test training
    labels = model.train_model(sample_data)
    assert len(labels) == len(sample_data), "The number of labels should match the data points."
    
    # Save and load model
    model.save_model("test_kmeans.pkl")
    model.load_model("test_kmeans.pkl")
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data), "The number of predictions should match the data points."

if __name__ == "__main__":
    pytest.main()
