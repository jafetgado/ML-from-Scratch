""" K-Nearest Neighbors algorithm """


import numpy as np
from abc import ABC, abstractmethod




class KNeighbors(ABC):
    """
    Base class for k-nearest neighbors algorithms.

    Attributes:
    - n_neighbors (int): Number of neighbors to use for prediction.
    - weights (str): Weight function used in prediction. Can be 'uniform' or 'distance'.
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Target values.
    - random_state (int): Random seed for reproducibility.
    - n_features_in_ (int): Number of features seen during fit.
    - n_samples_ (int): Number of samples seen during fit.

    Methods:
    - fit(X, y): Fit the model using X as training data and y as target values.
    - predict(X): Predict the target values for the provided data.
    """

    def __init__(self, n_neighbors=5, weights='uniform', random_state=None):
        """Initialize the KNeighbors model."""
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.random_state = random_state
        self.n_features_in_ = None
        self.n_samples_ = None


    def fit(self, X, y):
        """Fit the k-nearest neighbors model from the training dataset."""

        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        assert self.n_neighbors <= self.n_samples, \
            f"Expected n_neighbors <= n_samples, but got {self.n_neighbors} > {self.n_samples}"
        self.X_train = X
        self.y_train = y


    def _compute_distances(self, X):
        """Compute the Euclidean distances between X and training data."""
        
        # Check if number of features matches training data
        assert X.shape[1] == self.n_features_in_, \
            f"Expected {self.n_features_in_} features, got {X.shape[1]} features instead"
        
        # Compute squared distances without using loops
        X_squared = np.sum(X**2, axis=1, keepdims=True)  # Sum of squared elements
        X_train_squared = np.sum(self.X_train**2, axis=1)
        distances = X_squared + X_train_squared - 2 * np.dot(X, self.X_train.T)
        return np.sqrt(distances)


    def _get_neighbors(self, distances):
        """Get the indices of the k nearest neighbors for each test sample given the distances
        between test and training samples."""
        
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]


    def _get_weights(self, distances, neighbor_indices):
        """Compute weights for each neighbor based on the weight function."""

        if self.weights == 'uniform':
            return np.ones_like(neighbor_indices, dtype=float)
        elif self.weights == 'distance':
            # Get distances to k nearest neighbors
            neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
            # Avoid division by zero
            neighbor_distances = np.maximum(neighbor_distances, 1e-10)
            return 1.0 / neighbor_distances
        else:
            raise ValueError("weights must be either 'uniform' or 'distance'")


    @abstractmethod
    def predict(self, X):
        """Predict the target values for the provided data."""
        pass




class KNeighborsClassifier(KNeighbors):

    def predict(self, X):
        """Predict the class labels for the provided data."""
        
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Compute distances and get nearest neighbors
        distances = self._compute_distances(X)
        neighbor_indices = self._get_neighbors(distances)
        weights = self._get_weights(distances, neighbor_indices)

        # Get the classes of the nearest neighbors
        neighbor_classes = self.y_train[neighbor_indices]

        # For each test sample, compute weighted class counts
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Get unique classes and their weighted counts
            classes, counts = np.unique(neighbor_classes[i], return_counts=True)
            weighted_counts = np.zeros_like(counts, dtype=float)
            
            # Compute weighted counts for each class
            for j, cls in enumerate(classes):
                mask = neighbor_classes[i] == cls
                weighted_counts[j] = np.sum(weights[i][mask])
            
            # Predict the class with highest weighted count
            predictions[i] = classes[np.argmax(weighted_counts)]

        return predictions




class KNeighborsRegressor(KNeighbors):

    def predict(self, X):
        """Predict the target values for the provided data."""
        
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Compute distances and get nearest neighbors
        distances = self._compute_distances(X)
        neighbor_indices = self._get_neighbors(distances)
        weights = self._get_weights(distances, neighbor_indices)

        # Get the target values of the nearest neighbors
        neighbor_values = self.y_train[neighbor_indices]

        # For each test sample, compute weighted average
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=float)
        
        for i in range(n_samples):
            # Compute weighted average of neighbor values
            weighted_sum = np.sum(weights[i] * neighbor_values[i])
            total_weight = np.sum(weights[i])
            
            if total_weight > 0:
                predictions[i] = weighted_sum / total_weight
            else:
                predictions[i] = np.mean(neighbor_values[i])

        return predictions 