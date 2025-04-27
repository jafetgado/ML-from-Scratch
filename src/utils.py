"""Utility functions for machine learning algorithms."""


import numpy as np
import matplotlib.pyplot as plt




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




class SGDSolver:
    """
    Stochastic Gradient Descent solver with loss tracking and plotting.
    
    Attributes:
    -----------
    - learning_rate (float): Learning rate for gradient descent
    - n_epochs (int): Number of epochs to run
    - batch_size (int): Size of mini-batches
    - random_state (int): Random seed for reproducibility
    - losses (list): List of loss values for each epoch
    """
    
    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=32, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.losses = []
        

    def solve(self, X, y, loss_gradient, loss_fn=None):
        """
        Optimize weights using stochastic gradient descent.
        
        Parameters:
        -----------
        - X (ndarray): Training data with shape (n_samples, n_features)
        - y (ndarray): Target values with shape (n_samples,)
        - loss_gradient (callable): Function that computes the gradient of the loss function
        - loss_fn (callable): Optional function to compute loss for tracking
        
        Returns:
        --------
        - w (ndarray): Optimized weights with shape (n_features,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        w = np.zeros(n_features)  # Initialize weights
        self.losses = []  # Reset losses
        
        for epoch in range(self.n_epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Compute gradient and update weights in mini-batches
            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                gradient = loss_gradient(X_batch, y_batch, w)
                w -= self.learning_rate * gradient
            
            # Track loss if loss function provided
            if loss_fn is not None:
                loss = loss_fn(X, y, w)
                self.losses.append(loss)
            
        return w
        
        
    def plot_loss(self, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
        """Plot the loss over epochs."""
        
        if not self.losses:
            raise ValueError("No loss values to plot. Provide a loss_fn when calling solve().")
        plt.figure(figsize=(5,3))
        plt.plot(self.losses)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show(); plt.close()

