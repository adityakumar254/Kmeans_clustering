# ml_model/train.py

from ml_model.clustering import cluster_data

def deploy_clustering_model():
    """
    Train and deploy the clustering model.
    """
    # Call the clustering function to train the model
    clusters, model = cluster_data()
    print("Clustering model deployed successfully")
    return clusters, model

# This block is used for running the script directly, ensuring it's only executed when the script is run.
if __name__ == "__main__":
    deploy_clustering_model()


