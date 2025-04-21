# This is a sample Python script.
from EDA import DefectClustering


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Running Script, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # Example usage
    dataset_path = '/Users/isaacakakpo/Downloads/EnginerringProject3/capsule/'

    # Initialize with VGG16 (default)
    clustering = DefectClustering(
        dataset_path=dataset_path,
        n_clusters=5,  # For KMeans
        pca_components=50  # Number of PCA components
    )

    # Run with KMeans
    print("\n=== Running KMeans Clustering ===")
    kmeans_labels, kmeans_features = clustering.run_pipeline(method='kmeans')

    # Run with DBSCAN (auto-tuned eps)
    print("\n=== Running DBSCAN Clustering ===")
    dbscan_labels, dbscan_features = clustering.run_pipeline(
        method='dbscan',
        eps='auto',  # Auto-detect optimal eps
        min_samples=5  # Minimum points to form a cluster
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
