""" Trees and tree-based models """


import numpy as np
from abc import ABC, abstractmethod




class TreeNode:
    """
    Represents a single node in a decision tree.

    Attributes:
    - feature_index (int or None): Index of the feature to split on at this node.
        None if the node is a leaf.
    - threshold (float or None): Threshold value used to split the feature.
        None if the node is a leaf.
    - left (TreeNode or None): Left child node (samples that satisfy the split condition).
    - right (TreeNode or None): Right child node (samples that do not satisfy the condition).
    - value (int or float or None): Class label for classification or predicted value for 
        regression. Only set for leaf nodes.

    Methods:
    - is_leaf(): Returns True if the node is a leaf (i.e., has a predicted value).
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  
        self.threshold = threshold          
        self.left = left                    
        self.right = right                  
        self.value = value                  

    def is_leaf(self):
        return self.value is not None




class DecisionTreeBase(ABC):
    """
    Base class for decision trees.

    Attributes:
    - max_depth (int): Maximum depth allowed for the tree during training.
    - root (TreeNode or None): The root node of the trained decision tree.
    
    Methods:
    - fit(X, y): Builds the decision tree using training data X (features) and y (labels/values).
    - predict(X): Returns predicted labels/values for input data X.
    - print_tree(node=None, depth=0): Recursively prints the structure of the tree.
    """

    def __init__(self, max_depth=3):
        self.max_depth = max_depth   # Maximum depth of the tree
        self.root = None             # Root node of the tree


    @abstractmethod
    def _impurity(self, y):
        """Calculate the impurity of a node (to be implemented by child classes)."""
        pass


    @abstractmethod
    def _leaf_value(self, y):
        """Calculate the value for a leaf node (to be implemented by child classes)."""
        pass


    def _best_split(self, X, y):
        """Find the best feature and threshold to split the data."""
        
        best_impurity = float('inf')  # Initialize best impurity as infinity (to be minimized)
        best_feature = None           # Track best feature index
        best_threshold = None         # Track best threshold for splitting

        for idx in range(X.shape[1]):  # Iterate through each feature
            
            # Get feature values
            feature_values = X[:, idx]
            unique_values = np.unique(feature_values)
            
            # Skip if feature has only one unique value
            if len(unique_values) <= 1:
                continue
                  
            # If we have fewer than 100 unique values, use all of them
            if len(unique_values) <= 100:
                thresholds = unique_values
            else:
                # Otherwise, use 100 evenly spaced percentiles
                percentiles = np.linspace(0, 100, 100)
                thresholds = np.percentile(feature_values, percentiles)
            
            # Try each threshold
            for threshold in thresholds:
                # Split data based on threshold
                left_mask = X[:, idx] <= threshold
                right_mask = X[:, idx] > threshold
                y_left, y_right = y[left_mask], y[right_mask]
                
                # Skip if split would create empty nodes
                if len(y_left) == 0 or len(y_right) == 0:
                    continue # Skip invalid splits

                # Compute weighted impurity of the split
                impurity_left = self._impurity(y_left)
                impurity_right = self._impurity(y_right)
                weighted_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y)

                # Update if current split is better
                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = idx
                    best_threshold = threshold

        return best_feature, best_threshold  # Return best split

   
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        
        # If node is pure or max depth reached, return a leaf node
        if len(set(y)) == 1 or depth >= self.max_depth:
            leaf_value = self._leaf_value(y)
            return TreeNode(value=leaf_value)

        # Find best split
        feature_index, threshold = self._best_split(X, y)

        # If no split found, return leaf value from full dataset
        if feature_index is None:
            leaf_value = self._leaf_value(y)
            return TreeNode(value=leaf_value)

        # Split the data based on the best threshold
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return the decision node
        return TreeNode(feature_index, threshold, left_child, right_child)


    def fit(self, X, y):
        """Fit the decision tree to the training data."""
        self.root = self._build_tree(X, y, 0)  # Start at depth 0


    def _predict_one(self, x, node):
        """Predict one sample by recursively traversing the tree."""
        if node.is_leaf():
            return node.value  # Return predicted value if leaf
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)  # Go left
        else:
            return self._predict_one(x, node.right)  # Go right


    def predict(self, X):
        """Predict the target values for the input data."""
        return np.array([self._predict_one(x, self.root) for x in X])


    def print_tree(self, node=None, depth=0, feature_names=None):
        """Print the tree structure."""
        
        indent = "  " * depth  # Indentation for visualization
        if node is None:
            node = self.root
        if node.is_leaf():
            # Print leaf node with appropriate formatting
            if isinstance(node.value, (int, np.integer)):
                print(f"{indent}Leaf: {node.value}")  # Integer values (classification)
            else:
                print(f"{indent}Leaf: {node.value:.2f}")  # Float values (regression)
        else:
            # Print decision node
            if feature_names is not None:
                feature_name = feature_names[node.feature_index]
            else:
                feature_name = f"Feature[{node.feature_index}]"
            print(f"{indent}If {feature_name} <= {node.threshold:.2f}:")
            self.print_tree(node.left, depth + 1, feature_names)
            
            print(f"{indent}Else:")
            self.print_tree(node.right, depth + 1, feature_names)




class DecisionTreeClassifier(DecisionTreeBase):
    """Decision tree classifier using Gini impurity."""

    def _impurity(self, y):
        """Calculate the Gini impurity of a node."""
        _, counts = np.unique(y, return_counts=True)  # Count each class
        probs = counts / counts.sum()                 # Calculate class probabilities
        return 1 - np.sum(probs ** 2)                 # Gini formula

    def _leaf_value(self, y):
        """Return the most common class as the leaf value."""
        return np.bincount(y).argmax()




class DecisionTreeRegressor(DecisionTreeBase):
    """Decision tree regressor using mean squared error."""

    def _impurity(self, y):
        """Calculate the mean squared error of a node."""
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _leaf_value(self, y):
        """Return the mean value as the leaf value."""
        return np.mean(y)


