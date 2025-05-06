""" Trees and tree-based models """


import numpy as np
from abc import ABC, abstractmethod
from src import utils




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
    - feature_importances_ (np.ndarray): Array of feature importances computed after fitting.
    - random_state (int): Random seed for reproducibility.
    
    Methods:
    - fit(X, y): Builds the decision tree using training data X (features) and y (labels/values).
    - predict(X): Returns predicted labels/values for input data X.
    - print_tree(node=None, depth=0): Recursively prints the structure of the tree.
    - get_feature_importances(): Returns the computed feature importances.
    """
    
    def __init__(self, max_depth=3, random_state=None):
        self.max_depth = max_depth   # Maximum depth of the tree
        self.root = None             # Root node of the tree
        self.feature_importances_ = None  # Feature importances array
        self.n_features_ = None      # Number of features in the training data
        self.random_state = random_state


    @abstractmethod
    def _impurity(self, y):
        """Calculate the impurity of a node (to be implemented by child classes)."""
        pass


    @abstractmethod
    def _leaf_value(self, y):
        """Calculate the value for an impure leaf node (to be implemented by child classes)."""
        pass


    def _best_split(self, X, y):
        """Find the best feature and threshold to split the data."""
        best_impurity = float('inf')  # Initialize best impurity as infinity (to be minimized)
        best_feature = None           # Track best feature index
        best_threshold = None         # Track best threshold for splitting
        feature_indices = np.random.permutation(X.shape[1]) # Randomly shuffle feature indices

        for idx in feature_indices:  # Iterate through selected features in random order
            
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
                # Use 100 equi-distant bins between min and max values
                min_val, max_val = np.min(feature_values), np.max(feature_values)
                thresholds = np.linspace(min_val, max_val, 100)
            
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


    def _compute_feature_importance(self, node, total_impurity_reduction):
        """Recursively compute feature importance for each nodes"""
        if node.is_leaf():
            return total_impurity_reduction # nd.array of shape (n_features,)

        # Add the impurity reduction for this node's feature
        total_impurity_reduction[node.feature_index] += node.impurity_reduction

        # Recursively compute for child nodes
        total_impurity_reduction = self._compute_feature_importance(node.left, total_impurity_reduction)
        total_impurity_reduction = self._compute_feature_importance(node.right, total_impurity_reduction)
        
        return total_impurity_reduction


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

        # Calculate impurity reduction for this split
        parent_impurity = self._impurity(y)
        left_impurity = self._impurity(y[left_mask])
        right_impurity = self._impurity(y[right_mask])
        n_samples, n_left, n_right = len(y), len(y[left_mask]), len(y[right_mask])
        impurity_reduction = parent_impurity - (n_left/n_samples * left_impurity + n_right/n_samples * right_impurity)

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Create node with impurity reduction information
        node = TreeNode(feature_index, threshold, left_child, right_child)
        node.impurity_reduction = impurity_reduction
        
        return node


    def fit(self, X, y):
        """Fit the decision tree to the training data."""
        np.random.seed(self.random_state)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, 0)  # Start at depth 0
        
        # Initialize impurity reduction array with zeros
        total_impurity_reduction = np.zeros(self.n_features_)
        
        # Sum up impurity reductions over all nodes for all features
        total_impurity_reduction = self._compute_feature_importance(self.root, total_impurity_reduction) 
        
        # Feature importances are normalized impurity reductions
        self.feature_importances_ = total_impurity_reduction / np.sum(total_impurity_reduction)


    def get_feature_importances(self):
        """Return the computed feature importances."""
        if self.feature_importances_ is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        return self.feature_importances_


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
        """Return the Gini impurity of a node."""
        return utils.gini_importance(y)


    def _leaf_value(self, y):
        """Return the most common class as the leaf value."""
        return np.bincount(y).argmax()




class DecisionTreeRegressor(DecisionTreeBase):
    """Decision tree regressor using mean squared error."""

    def _impurity(self, y):
        """Calculate the mean squared error of a node as the impurity."""
        return utils.mean_squared_deviation(y)


    def _leaf_value(self, y):
        """Return the mean value as the leaf value."""
        return np.mean(y)




class RandomForestBase(ABC):
    """
    Base class for random forests.

    Attributes:
    - n_estimators (int): Number of trees in the forest.
    - max_depth (int): Maximum depth allowed for each tree during training.
    - random_state (int): Random seed for reproducibility.
    - estimators_ (list): List of trained decision trees.
    - feature_importances_ (np.ndarray): Array of feature importances computed after fitting.
    
    Methods:
    - fit(X, y): Builds the random forest using training data X (features) and y (labels/values).
    - predict(X): Returns predicted labels/values for input data X.
    - get_feature_importances(): Returns the computed feature importances.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.feature_importances_ = None


    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample of the data."""
        n_samples = X.shape[0]
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[sample_indices], y[sample_indices]
    
    
    @abstractmethod
    def _make_tree(self):
        """Create a new decision tree (to be implemented by child classes)."""
        pass


    @abstractmethod
    def _predict(self, X):
        """Predict the target values for the input data (to be implemented by child classes)."""
        pass


    def fit(self, X, y):
        """Fit the random forest to the training data."""
        np.random.seed(self.random_state)
        self.n_features_ = X.shape[1]
        
        # Initialize feature importances
        total_importances = np.zeros(self.n_features_)
        
        # Grow trees
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Create and fit tree
            tree = self._make_tree()
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)
            
            # Accumulate feature importances
            total_importances += tree.get_feature_importances()
            
        # Average feature importances across all trees
        self.feature_importances_ = total_importances / self.n_estimators


    def get_feature_importances(self):
        """Return the computed feature importances."""
        if self.feature_importances_ is None:
            raise ValueError("Forest has not been fitted yet. Call fit() first.")
        return self.feature_importances_




class RandomForestClassifier(RandomForestBase):
    """Random forest for classification tasks."""
    
    def _make_tree(self):
        """Create a new decision tree classifier."""
        return DecisionTreeClassifier(max_depth=self.max_depth)
        
        
    def predict(self, X):
        """Predict class labels for input data."""
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        ypred = np.array([np.bincount(pred).argmax() for pred in predictions.T])
        return ypred




class RandomForestRegressor(RandomForestBase):
    """Random forest for regression tasks."""
    
    def _make_tree(self):
        """Create a new decision tree regressor."""
        return DecisionTreeRegressor(max_depth=self.max_depth)


    def predict(self, X):
        """Predict target values for input data."""
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)