import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from sklearn.cluster import KMeans, DBSCAN, OPTICS  # Added OPTICS
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE  # Added for alternative visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import warnings


class DefectClustering:
    def __init__(self, dataset_path, n_clusters=5, pca_components=50):
        """
        Enhanced defect clustering with multiple model support and improved clustering.

        Args:
            dataset_path (str): Path to dataset directory
            n_clusters (int): Default number of clusters for KMeans
            pca_components (int): Number of PCA components for dimensionality reduction
        """
        self.dataset_path = dataset_path
        self.n_clusters = n_clusters
        self.pca_components = pca_components

        # Suppress unnecessary warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        # Initialize with VGG16 by default
        self.initialize_model('vgg16')

        # Setup directory paths with validation
        self._validate_and_set_directories()

        # Initialize clustering parameters
        self._init_clustering_params()

        print("Defect clustering initialized successfully")

    def _validate_and_set_directories(self):
        """Validate and set directory paths with error handling."""
        required_dirs = ['train', 'test', 'ground_truth']
        self.train_dir = os.path.join(self.dataset_path, 'train')
        self.test_dir = os.path.join(self.dataset_path, 'test')
        self.ground_truth_dir = os.path.join(self.dataset_path, 'ground_truth')

        for dir_path, dir_name in zip(
                [self.train_dir, self.test_dir, self.ground_truth_dir],
                required_dirs
        ):
            if not os.path.isdir(dir_path):
                raise ValueError(f"Required directory '{dir_name}' not found at {dir_path}")

    def _init_clustering_params(self):
        """Initialize default clustering parameters."""
        self.clustering_params = {
            'kmeans': {
                'init': 'k-means++',
                'n_init': 10,
                'random_state': 42
            },
            'dbscan': {
                'min_samples': 5,
                'metric': 'euclidean',
                'eps': 'auto'
            },
            'optics': {
                'min_samples': 5,
                'metric': 'euclidean',
                'xi': 0.05
            }
        }

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
        """Load and preprocess image for the selected model with error handling."""
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return self.preprocess_input(img_array)
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

    def extract_features_from_directory(self, directory):
        """Robust feature extraction with progress tracking and error handling."""
        features = []
        image_paths = []
        skipped_images = 0

        for defect_type in os.listdir(directory):
            defect_dir = os.path.join(directory, defect_type)
            if not os.path.isdir(defect_dir):
                continue

            for img_name in tqdm(os.listdir(defect_dir),
                                 desc=f"Processing {defect_type}",
                                 unit="image"):
                img_path = os.path.join(defect_dir, img_name)
                if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_array = self.load_and_preprocess_image(img_path)
                if img_array is None:
                    skipped_images += 1
                    continue

                try:
                    feature_vector = self.model.predict(img_array, verbose=0)
                    features.append(feature_vector.flatten())
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {str(e)}")
                    skipped_images += 1

        if skipped_images > 0:
            print(f"Warning: Skipped {skipped_images} images due to processing errors")

        return np.array(features), image_paths

    def reduce_dimensions(self, features, method='pca'):
        """
        Dimensionality reduction with multiple method support.

        Args:
            features: Input feature matrix
            method: 'pca' or 'tsne'
        Returns:
            Reduced feature matrix
        """
        if method == 'pca':
            reducer = PCA(n_components=self.pca_components, random_state=42)
            reduced_features = reducer.fit_transform(features)
            print(f"PCA explained variance ratio: {sum(reducer.explained_variance_ratio_):.2f}")
        elif method == 'tsne':
            # First reduce with PCA for efficiency
            pca = PCA(n_components=50, random_state=42)
            pca_features = pca.fit_transform(features)

            # Then apply t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            reduced_features = reducer.fit_transform(pca_features)
            print("t-SNE dimensionality reduction completed")
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return StandardScaler().fit_transform(reduced_features)

    def find_optimal_eps(self, features, min_samples=5, plot=True):
        """
        Enhanced automatic eps detection with multiple methods.

        Args:
            features: Input feature matrix
            min_samples: min_samples parameter for DBSCAN
            plot: Whether to show diagnostic plots
        Returns:
            Optimal eps value
        """
        # Method 1: Knee point detection
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features)
        distances, _ = nbrs.kneighbors(features)
        k_distances = np.sort(distances[:, -1])

        # Find knee point using maximum curvature
        differences = np.diff(k_distances)
        knee_point = np.argmax(differences) + 1
        eps_knee = k_distances[knee_point]

        # Method 2: Percentile-based approach
        eps_percentile = np.percentile(k_distances, 95)

        # Choose the more conservative (smaller) eps
        optimal_eps = min(eps_knee, eps_percentile)

        if plot:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
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

            plt.subplot(1, 2, 2)
            sns.histplot(k_distances, kde=True)
            plt.axvline(x=optimal_eps, color='r', linestyle='--')
            plt.title('Distance Distribution')
            plt.xlabel('Distance')
            plt.tight_layout()
            plt.show()

        return optimal_eps

    def cluster_data(self, features, method='kmeans', **kwargs):
        """
        Unified clustering interface with multiple algorithm support.

        Args:
            features: Input feature matrix
            method: 'kmeans', 'dbscan', or 'optics'
            **kwargs: Algorithm-specific parameters
        Returns:
            Tuple of (labels, cluster_metrics, cluster_distribution)
        """
        method = method.lower()

        # Normalize features
        features = StandardScaler().fit_transform(features)

        if method == 'kmeans':
            params = {**self.clustering_params['kmeans'], **kwargs}
            clusterer = KMeans(n_clusters=self.n_clusters, **params)
            labels = clusterer.fit_predict(features)

        elif method == 'dbscan':
            params = {**self.clustering_params['dbscan'], **kwargs}
            if params['eps'] == 'auto':
                params['eps'] = self.find_optimal_eps(features, params['min_samples'])
            clusterer = DBSCAN(**params)
            labels = clusterer.fit_predict(features)

        elif method == 'optics':
            params = {**self.clustering_params['optics'], **kwargs}
            clusterer = OPTICS(**params)
            labels = clusterer.fit_predict(features)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Calculate metrics
        metrics = self._calculate_cluster_metrics(features, labels)
        dist_table = self._create_cluster_distribution(labels)

        return labels, metrics, dist_table

    def _calculate_cluster_metrics(self, features, labels):
        """Calculate comprehensive clustering metrics."""
        if len(set(labels)) < 2:
            return {
                'Silhouette Score': np.nan,
                'Calinski-Harabasz': np.nan,
                'Davies-Bouldin': np.nan
            }

        return {
            'Silhouette Score': silhouette_score(features, labels),
            'Calinski-Harabasz': calinski_harabasz_score(features, labels),
            'Davies-Bouldin': davies_bouldin_score(features, labels)
        }

    def _create_cluster_distribution(self, labels):
        """Create formatted cluster distribution table."""
        unique, counts = np.unique(labels, return_counts=True)
        return pd.DataFrame({
            'Cluster': unique,
            'Samples': counts,
            'Percentage': [f"{c / len(labels) * 100:.1f}%" for c in counts]
        })

    def visualize_clusters(self, features, labels, method='PCA'):
        """
        Enhanced cluster visualization with multiple view options.

        Args:
            features: Input feature matrix
            labels: Cluster labels
            method: 'PCA' or 'TSNE' for visualization
        """
        if method == 'TSNE':
            vis_features = self.reduce_dimensions(features, 'tsne')
            method_name = 't-SNE'
        else:
            vis_features = features[:, :2]  # Use first two PCA components
            method_name = 'PCA'

        plt.figure(figsize=(14, 10))

        # Create colormap that highlights noise
        if -1 in labels:
            n_clusters = len(set(labels)) - 1
            colors = sns.color_palette('viridis', n_clusters)
            colors.insert(0, (1, 0, 0))  # Red for noise
            cmap = ListedColormap(colors)
        else:
            cmap = 'viridis'

        # Main scatter plot
        scatter = plt.scatter(
            vis_features[:, 0], vis_features[:, 1],
            c=labels, cmap=cmap,
            s=50, alpha=0.7,
            edgecolor='w', linewidth=0.5
        )

        # Add cluster centers for KMeans if applicable
        if len(set(labels)) > 1 and -1 not in labels:
            centers = np.array([vis_features[labels == i].mean(axis=0)
                                for i in range(len(set(labels)))])
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
            f"Defect Clustering ({model_name})\n"
            f"Visualization: {method_name}, Samples: {len(labels)}",
            pad=20
        )
        plt.xlabel(f"{method_name} Component 1")
        plt.ylabel(f"{method_name} Component 2")
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self, clustering_method='kmeans', reduction_method='pca', **kwargs):
        """
        Complete analysis pipeline with flexible configuration.

        Args:
            clustering_method: 'kmeans', 'dbscan', or 'optics'
            reduction_method: 'pca' or 'tsne'
            **kwargs: Additional clustering parameters
        Returns:
            Tuple of (labels, metrics, cluster_distribution)
        """
        print("\n=== Starting Full Analysis ===")

        # 1. Feature extraction
        print("\n[1/4] Extracting features...")
        train_features, _ = self.extract_features_from_directory(self.train_dir)
        test_features, _ = self.extract_features_from_directory(self.test_dir)
        all_features = np.vstack([train_features, test_features])

        # 2. Dimensionality reduction
        print("\n[2/4] Reducing dimensions...")
        reduced_features = self.reduce_dimensions(all_features, reduction_method)

        # 3. Clustering
        print(f"\n[3/4] Running {clustering_method.upper()} clustering...")
        labels, metrics, dist_table = self.cluster_data(
            reduced_features,
            method=clustering_method,
            **kwargs
        )

        # 4. Visualization and results
        print("\n[4/4] Visualizing results...")
        self.visualize_clusters(reduced_features, labels, reduction_method)

        # Print results
        print("\n=== Clustering Results ===")
        print(f"\nMethod: {clustering_method.upper()}")
        print("\nMetrics:")
        print(tabulate(pd.DataFrame([metrics]), headers='keys', tablefmt='grid'))
        print("\nCluster Distribution:")
        print(tabulate(dist_table, headers='keys', tablefmt='grid'))

        return labels, metrics, dist_table


# Example usage
if __name__ == '__main__':
    dataset_path = '/Users/isaacakakpo/Downloads/EnginerringProject3/capsule/'

    # Initialize analyzer
    analyzer = DefectClustering(
        dataset_path=dataset_path,
        n_clusters=5,
        pca_components=50
    )

    # Option 1: Run KMeans with PCA
    print("\nRunning KMeans analysis...")
    kmeans_labels, kmeans_metrics, kmeans_dist = analyzer.run_full_analysis(
        clustering_method='kmeans',
        reduction_method='pca'
    )

    # Option 2: Run DBSCAN with auto-tuned eps
    print("\nRunning DBSCAN analysis...")
    dbscan_labels, dbscan_metrics, dbscan_dist = analyzer.run_full_analysis(
        clustering_method='dbscan',
        reduction_method='pca',
        eps='auto',
        min_samples=5
    )

    # Option 3: Run OPTICS with t-SNE visualization
    print("\nRunning OPTICS analysis...")
    optics_labels, optics_metrics, optics_dist = analyzer.run_full_analysis(
        clustering_method='optics',
        reduction_method='tsne',
        min_samples=5,
        xi=0.05
    )