"""
losses.py

A collection of loss functions and their gradients for use in neural networks.

This module includes a class that provides static methods for various loss functions,
such as Mean Squared Error, Binary Cross-Entropy, and Categorical Cross-Entropy.
These loss functions are essential for training neural networks, allowing them to measure
the difference between predicted and actual values and guide the optimization process.

Usage:
    To use this module, import the Loss class and call the desired loss function 
    or its gradient as a static method. For example:

        import loss
        mse = loss.Loss.mean_squared_error(y_true, y_pred)
"""

import numpy as np

class Loss:
    """
    A collection of loss functions and their gradients for use in neural networks.

    This class provides static methods for computing common loss functions 
    and their derivatives. These functions are crucial in training models 
    by providing a measure of the error between predicted and actual outputs.

    Methods:
        mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Computes the Mean Squared Error (MSE) loss.

        mean_squared_error_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            Computes the derivative of the MSE loss.

        binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Computes the Binary Cross-Entropy loss.

        binary_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            Computes the derivative of the Binary Cross-Entropy loss.

        categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Computes the Categorical Cross-Entropy loss.

        categorical_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            Computes the derivative of the Categorical Cross-Entropy loss.
    """

    @staticmethod
    def mean_squared_error(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Computes the Mean Squared Error (MSE) loss.

        MSE measures the average of the squares of the errors—that is, the average squared 
        difference between the estimated values (predictions) and the actual value.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The computed Mean Squared Error.
        """
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mean_squared_error_gradient(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Computes the gradient of the Mean Squared Error loss.

        The gradient of MSE is used during backpropagation to adjust weights.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: The gradient of the Mean Squared Error with respect to the predictions.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Computes the Binary Cross-Entropy loss.

        Binary Cross-Entropy measures the performance of a classification model whose output 
        is a probability value between 0 and 1.

        Args:
            y_true (np.ndarray): True binary labels (0 or 1).
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: The computed Binary Cross-Entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_gradient(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Computes the derivative of the Binary Cross-Entropy loss.

        The gradient is used during backpropagation to update weights based on the loss.

        Args:
            y_true (np.ndarray): True binary labels (0 or 1).
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            np.ndarray: The gradient of the Binary Cross-Entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def categorical_cross_entropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Computes the Categorical Cross-Entropy loss.

        Categorical Cross-Entropy measures the performance of a classification model whose output 
        is a probability value distributed across multiple classes.

        Args:
            y_true (np.ndarray): True labels in one-hot encoded format.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: The computed Categorical Cross-Entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def categorical_cross_entropy_gradient(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Computes the derivative of the Categorical Cross-Entropy loss.

        The gradient is used during backpropagation to update weights in multi-class models.

        Args:
            y_true (np.ndarray): True labels in one-hot encoded format.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            np.ndarray: The gradient of the Categorical Cross-Entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_true / y_pred) / y_pred.shape[0]