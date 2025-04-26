"""Utility functions for machine learning algorithms."""


import numpy as np




def gini_importance(y):
    """Calculate the Gini importance of a node."""
    _, counts = np.unique(y, return_counts=True)  # Count each class
    probs = counts / counts.sum()                 # Calculate class probabilities
    return 1 - np.sum(probs ** 2)                 # Gini formula 




def mean_squared_deviation(y, root=False):
    """Calculate the mean squared deviation of a node."""
    if len(y) == 0:
        return 0
    dev = np.mean((y - np.mean(y)) ** 2)
    if root:
        dev = np.sqrt(dev)
    return dev

