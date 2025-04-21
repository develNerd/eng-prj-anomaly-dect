import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate


class DefectClustering:
    def __init__(self, dataset_path, n_clusters=5, pca_components=50):
        """
        Initialize defect clustering with flexible model selection and preprocessing.

        Args:
            dataset_path (str): Path to dataset directory
            n_clusters (int): Default number of clusters for KMeans
            pca_components (int): Number of PCA components for dimensionality reduction
        """
        self.dataset_path = dataset_path
        self.n_clusters = n_clusters
        self.pca_components = pca_components

        # Initialize model (default to VGG16)
        self.initialize_model('vgg16')

        # Setup directory paths
        self.train_dir = os.path.join(dataset_path, 'train')
        self.test_dir = os.path.join(dataset_path, 'test')
        self.ground_truth_dir = os.path.join(dataset_path, 'ground_truth')
        print("Directories initialized successfully")

    def initialize_model(self, model_name='vgg16'):
        """Initialize the selected pretrained model with proper preprocessing."""
        model_name = model_name.lower()
        if model_name == 'resnet50':
            self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = resnet_preprocess
            print("Using ResNet50 for feature extraction")
        else:  # Default to VGG16
            self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = vgg_preprocess
            print("Using VGG16 for feature extraction")

    def load_and_preprocess_image(self, img_path):
        """Load and preprocess image for the selected model."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return self.preprocess_input(img_array)

    def extract_features_from_directory(self, directory):
        """Extract features for all images in a directory with progress tracking."""
        features = []
        image_paths = []

        for defect_type in os.listdir(directory):
            defect_dir = os.path.join(directory, defect_type)
            if os.path.isdir(defect_dir):
                for img_name in tqdm(os.listdir(defect_dir),
                                     desc=f"Processing {defect_type}",
                                     unit="image"):
                    img_path = os.path.join(defect_dir, img_name)
                    if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img_array = self.load_and_preprocess_image(img_path)
                            feature_vector = self.model.predict(img_array, verbose=0)
                            features.append(feature_vector.flatten())
                            image_paths.append(img_path)
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")

        return np.array(features), image_paths

    def analyze_clusters(self, features, labels):
        """Generate comprehensive cluster analysis."""
        # Calculate evaluation metrics
        metrics = {
            'Silhouette Score': silhouette_score(features, labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(features, labels),
            'Davies-Bouldin Index': davies_bouldin_score(features, labels)
        }

        # Cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_dist = pd.DataFrame({
            'Cluster': unique_labels,
            'Samples': counts,
            'Percentage': [f"{(count / len(labels)) * 100:.1f}%" for count in counts]
        })

        return metrics, cluster_dist

    def plot_clusters(self, features, labels, method='KMeans'):
        """Enhanced visualization of clustering results."""
        plt.figure(figsize=(14, 10))

        # Create custom colormap that highlights noise (if present)
        if -1 in labels:
            n_clusters = len(set(labels)) - 1
            colors = sns.color_palette('viridis', n_clusters)
            colors.insert(0, (1, 0, 0))  # Red for noise
            cmap = ListedColormap(colors)
        else:
            cmap = 'viridis'

        # Main scatter plot
        scatter = plt.scatter(
            features[:, 0], features[:, 1],
            c=labels, cmap=cmap,
            s=50, alpha=0.7,
            edgecolor='w', linewidth=0.5
        )

        # Add cluster centers for KMeans
        if method == 'KMeans':
            centers = np.array([features[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            plt.scatter(
                centers[:, 0], centers[:, 1],
                c='red', marker='X',
                s=200, edgecolor='k',
                linewidth=1, label='Cluster Centers'
            )
            plt.legend()

        # Plot formatting
        model_name = 'ResNet50' if 'resnet' in str(self.model).lower() else 'VGG16'
        plt.title(
            f"Defect Clustering ({model_name} + {method})\n"
            f"PCA Components: {self.pca_components}, Samples: {len(labels)}",
            pad=20
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Cluster ID")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def find_optimal_eps(self, features, min_samples, plot=True):
        """
        Automatically determine optimal eps using the knee method.

        Args:
            features: Input feature matrix
            min_samples: min_samples parameter for DBSCAN
            plot: Whether to show the knee plot
        Returns:
            Optimal eps value
        """
        # Compute distances to k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features)
        distances, _ = nbrs.kneighbors(features)

        # Sort distances
        k_distances = np.sort(distances[:, -1])

        # Find the knee point (max curvature)
        differences = np.diff(k_distances)
        knee_point = np.argmax(differences) + 1  # +1 because diff reduces array size
        optimal_eps = k_distances[knee_point]

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(k_distances, 'b-', label='k-distance')
            plt.axhline(y=optimal_eps, color='r', linestyle='--',
                        label=f'Suggested eps: {optimal_eps:.2f}')
            plt.axvline(x=knee_point, color='g', linestyle=':',
                        label=f'Knee point')
            plt.xlabel('Points sorted by distance')
            plt.ylabel(f'{min_samples}-th nearest neighbor distance')
            plt.title('Knee Plot for Optimal Eps Selection')
            plt.legend()
            plt.grid()
            plt.show()

        return optimal_eps

    def cluster_with_kmeans(self, features):
        """Perform KMeans clustering with comprehensive analysis."""
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Cluster with KMeans
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(features_normalized)

        # Analyze results
        metrics, cluster_dist = self.analyze_clusters(features_normalized, labels)

        # Visualize
        self.plot_clusters(features_normalized, labels, 'KMeans')

        # Print results
        print("\nKMeans Clustering Results:")
        print(tabulate(pd.DataFrame([metrics]), headers='keys', tablefmt='grid'))
        print("\nCluster Distribution:")
        print(tabulate(cluster_dist, headers='keys', tablefmt='grid'))

        return labels, features_normalized

    def cluster_with_dbscan(self, features, eps='auto', min_samples=5):
        """Perform DBSCAN clustering with comprehensive analysis."""
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Auto-tune eps if needed
        if eps == 'auto':
            eps = self.find_optimal_eps(features_normalized, min_samples)

        # Cluster with DBSCAN
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels = dbscan.fit_predict(features_normalized)

        # Analyze results
        metrics, cluster_dist = self.analyze_clusters(features_normalized, labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Visualize
        self.plot_clusters(features_normalized, labels, 'DBSCAN')

        # Print results
        print("\nDBSCAN Clustering Results:")
        print(f"- Estimated eps: {eps:.2f}")
        print(f"- Min samples: {min_samples}")
        print(f"- Number of clusters: {n_clusters}")
        print(f"- Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")
        print("\nCluster Metrics:")
        print(tabulate(pd.DataFrame([metrics]), headers='keys', tablefmt='grid'))
        print("\nCluster Distribution:")
        print(tabulate(cluster_dist, headers='keys', tablefmt='grid'))

        return labels, features_normalized

    def run_pipeline(self, method='kmeans', **kwargs):
        """
        Complete clustering pipeline with flexible method selection.

        Args:
            method (str): 'kmeans' or 'dbscan'
            **kwargs: Additional parameters for clustering method
        """
        # Feature extraction
        print("\nExtracting features from training set...")
        train_features, _ = self.extract_features_from_directory(self.train_dir)
        print("Extracting features from test set...")
        test_features, _ = self.extract_features_from_directory(self.test_dir)
        all_features = np.vstack([train_features, test_features])

        # Dimensionality reduction
        print(f"\nReducing dimensions with PCA (n_components={self.pca_components})...")
        pca = PCA(n_components=self.pca_components, random_state=42)
        reduced_features = pca.fit_transform(all_features)
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")

        # Clustering
        if method.lower() == 'dbscan':
            eps = kwargs.get('eps', 'auto')
            min_samples = kwargs.get('min_samples', 5)
            print("\nRunning DBSCAN clustering...")
            return self.cluster_with_dbscan(reduced_features, eps, min_samples)
        else:
            print("\nRunning KMeans clustering...")
            return self.cluster_with_kmeans(reduced_features)


