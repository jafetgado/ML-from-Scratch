"""Linear models for regression and classification."""


import numpy as np
from abc import ABC, abstractmethod
from src.utils import SGDSolver


class LinearModelBase(ABC):
    """
    Base class for linear models.
    
    Attributes:
    -----------
    - learning_rate (float): Learning rate for gradient descent
    - n_epochs (int): Number of epochs for training
    - batch_size (int): Size of mini-batches
    - random_state (int): Random seed for reproducibility
    - coef_ (ndarray): Model coefficients
    - intercept_ (float): Model intercept
    - solver (SGDSolver): SGD solver instance
    """
    
    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=32, random_state=None):
        """Initialize LinearModelBase."""
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.solver = None  # Will be initialized in fit()


    @abstractmethod
    def _loss_gradient(self, X, y, w):
        """Compute the gradient of the loss function."""
        pass
        

    @abstractmethod
    def _loss(self, X, y, w):
        """Compute the loss function."""
        pass


    def fit(self, X, y):
        """Fit the model to the training data."""
        # Add bias term
        X_bias = np.ones((X.shape[0], 1))
        X_bias = np.concatenate([X, X_bias], axis=1)
        
        # Initialize weights (including bias)
        w = np.zeros(X_bias.shape[1])
        
        # Create solver instance
        self.solver = SGDSolver(
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            random_state=self.random_state
        )
        
        # Optimize weights using SGD
        w = self.solver.solve(
            X_bias, y,
            loss_gradient=self._loss_gradient,
            loss_fn=self._loss
        )
        
        # Split weights into intercept and coefficients
        self.intercept_ = w[0]
        self.coef_ = w[1:]


    def plot_loss(self, title=None, xlabel="Epoch", ylabel="Loss"):
        """Plot the training loss over epochs."""
        if self.solver is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if title is None:
            title = f"{self.__class__.__name__} Training Loss"
        self.solver.plot_loss(title=title, xlabel=xlabel, ylabel=ylabel)
        
    def predict(self, X):
        """Make predictions for input data."""
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return X @ self.coef_ + self.intercept_




class LinearRegression(LinearModelBase):
    """Linear regression without regularization."""
    
    def _loss_gradient(self, X, y, w):
        """Compute gradient of MSE loss."""
        n_samples = X.shape[0]
        y_pred = X @ w
        return (2/n_samples) * X.T @ (y_pred - y)
        
        
    def _loss(self, X, y, w):
        """Compute MSE loss."""
        n_samples = X.shape[0]
        y_pred = X @ w
        return np.mean((y_pred - y) ** 2)


class Ridge(LinearModelBase):
    """Linear regression with L2 regularization."""
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def _loss_gradient(self, X, y, w):
        """Compute gradient of MSE loss with L2 regularization."""
        n_samples = X.shape[0]
        y_pred = X @ w
        # Don't regularize the bias term
        reg_term = np.zeros_like(w)
        reg_term[1:] = self.alpha * w[1:]
        return (2/n_samples) * X.T @ (y_pred - y) + 2 * reg_term
        
    def _loss(self, X, y, w):
        """Compute MSE loss with L2 regularization."""
        n_samples = X.shape[0]
        y_pred = X @ w
        mse = np.mean((y_pred - y) ** 2)
        reg = self.alpha * np.sum(w[1:] ** 2)  # Don't regularize bias
        return mse + reg


class Lasso(LinearModelBase):
    """Linear regression with L1 regularization."""
    
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def _loss_gradient(self, X, y, w):
        """Compute gradient of MSE loss with L1 regularization."""
        n_samples = X.shape[0]
        y_pred = X @ w
        # Don't regularize the bias term
        reg_term = np.zeros_like(w)
        reg_term[1:] = self.alpha * np.sign(w[1:])
        return (2/n_samples) * X.T @ (y_pred - y) + reg_term
        
    def _loss(self, X, y, w):
        """Compute MSE loss with L1 regularization."""
        n_samples = X.shape[0]
        y_pred = X @ w
        mse = np.mean((y_pred - y) ** 2)
        reg = self.alpha * np.sum(np.abs(w[1:]))  # Don't regularize bias
        return mse + reg


class ElasticNet(LinearModelBase):
    """Linear regression with both L1 and L2 regularization."""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def _loss_gradient(self, X, y, w):
        """Compute gradient of MSE loss with elastic net regularization."""
        n_samples = X.shape[0]
        y_pred = X @ w
        
        # Don't regularize the bias term
        reg_term = np.zeros_like(w)
        reg_term[1:] = (
            self.alpha * self.l1_ratio * np.sign(w[1:]) +  # L1 term
            self.alpha * (1 - self.l1_ratio) * w[1:]       # L2 term
        )
        
        return (2/n_samples) * X.T @ (y_pred - y) + reg_term
        
    def _loss(self, X, y, w):
        """Compute MSE loss with elastic net regularization."""
        n_samples = X.shape[0]
        y_pred = X @ w
        mse = np.mean((y_pred - y) ** 2)
        
        # Don't regularize bias term
        l1_reg = self.alpha * self.l1_ratio * np.sum(np.abs(w[1:]))
        l2_reg = self.alpha * (1 - self.l1_ratio) * np.sum(w[1:] ** 2)
        
        return mse + l1_reg + l2_reg


class LogisticRegression(LinearModelBase):
    """Logistic regression for binary classification."""
    
    def _sigmoid(self, z):
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-z))
        
    def _loss_gradient(self, X, y, w):
        """Compute gradient of logistic loss."""
        n_samples = X.shape[0]
        y_pred = self._sigmoid(X @ w)
        return (1/n_samples) * X.T @ (y_pred - y)
        
    def _loss(self, X, y, w):
        """Compute logistic loss."""
        n_samples = X.shape[0]
        y_pred = self._sigmoid(X @ w)
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        return -(1/n_samples) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._sigmoid(X @ self.coef_ + self.intercept_)
        
    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int) 