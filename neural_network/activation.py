"""
activation.py

A collection of activation functions and their gradients for use in neural networks.

This module includes a class that provides static methods for common activation functions, 
including sigmoid, ReLU (Rectified Linear Unit), and hyperbolic tangent (tanh). 
These functions, along with their gradients, are essential for introducing 
non-linearity into neural networks, allowing them to learn complex patterns 
and make more accurate predictions.

Usage:
    To use this module, import the Activation class and call the desired activation 
    function or its gradient as a static method. For example:

        import activation
        output = activation.Activation.sigmoid(input_array)

Functions include:
- Activation.sigmoid
- Activation.sigmoid_gradient
- Activation.relu
- Activation.relu_gradient
- Activation.tanh
- Activation.tanh_gradient
- Activation.softmax
- Activation.softmax_gradient
"""

import numpy as np

class Activation:
    """
    A collection of activation functions and their gradients for use in neural networks.

    This class provides static methods for common activation functions, 
    including sigmoid, ReLU, and hyperbolic tangent (tanh), along with 
    their respective gradients. These functions are essential in 
    introducing non-linearity into the network, enabling it to learn complex patterns.

    Methods:
        sigmoid(x: np.ndarray) -> np.ndarray:
            Applies the sigmoid activation function to the input array.

        sigmoid_gradient(x: np.ndarray) -> np.ndarray:
            Computes the gradient of the sigmoid function.

        relu(x: np.ndarray) -> np.ndarray:
            Applies the ReLU (Rectified Linear Unit) activation function.

        relu_gradient(x: np.ndarray) -> np.ndarray:
            Computes the gradient of the ReLU function.

        tanh(x: np.ndarray) -> np.ndarray:
            Applies the hyperbolic tangent activation function.

        tanh_gradient(x: np.ndarray) -> np.ndarray:
            Computes the gradient of the tanh function.

        softmax(x: np.ndarray) -> np.ndarray:
            Applies the softmax activation function.

        softmax_gradient(x: np.ndarray) -> np.ndarray:
            Computes the gradient of the softmax function.
    """

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid activation function.

        The sigmoid function maps input values to the range (0, 1), making it useful for
        models that need to output probabilities.

        Args:
            x (np.ndarray): Input array to which the sigmoid function is applied.

        Returns:
            np.ndarray: Output array where each element is the sigmoid of the corresponding 
                        element in the input array.
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_gradient(x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the sigmoid function.

        The gradient can be used during backpropagation to adjust weights in neural networks.

        Args:
            x (np.ndarray): Input array, typically the output of the sigmoid function.

        Returns:
            np.ndarray: Output array containing the gradient of the sigmoid function at 
                        each input point.
        """
        return x * (1 - x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU (Rectified Linear Unit) activation function.

        ReLU returns the input value if it is greater than zero, otherwise it returns zero.
        This function helps mitigate the vanishing gradient problem and allows models to learn 
        faster and perform better.

        Args:
            x (np.ndarray): Input array to which the ReLU function is applied.

        Returns:
            np.ndarray: Output array where each element is the result of applying 
                        the ReLU function to the corresponding element in the input array.
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_gradient(x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the ReLU function.

        The gradient is used during backpropagation to update weights based on the ReLU output.

        Args:
            x (np.ndarray): Input array, where the ReLU function was applied.

        Returns:
            np.ndarray: Output array containing the gradient of the ReLU function, 
                        which is 1 for positive inputs and 0 for non-positive inputs.
        """
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Applies the hyperbolic tangent activation function.

        The tanh function maps input values to the range (-1, 1), providing a symmetric output
        that helps improve convergence in training deep networks.

        Args:
            x (np.ndarray): Input array to which the tanh function is applied.

        Returns:
            np.ndarray: Output array where each element is the result of applying 
                        the tanh function to the corresponding element in the input array.
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_gradient(x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the hyperbolic tangent function.

        The gradient is used during backpropagation to update weights in neural networks.

        Args:
            x (np.ndarray): Input array, typically the output of the tanh function.

        Returns:
            np.ndarray: Output array containing the gradient of the tanh function at 
                        each input point.
        """
        return 1 - x ** 2
    
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function.

        The softmax function is used in multi-class classification problems to convert
        the output of a neural network into probabilities.

        Args:
            x (np.ndarray): Input array to which the softmax function is applied.

        Returns:
            np.ndarray: Output array where each element is the result of applying 
                        the softmax function to the corresponding element in the input array.
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def softmax_gradient(x: np.ndarray) -> int:
        """
        Computes the gradient of the softmax function.

        Args:
            x (np.ndarray): Input array, typically the output of the softmax function.

        Returns:
            np.ndarray: An array of ones with the same shape as the input.
        """
        return np.ones_like(x)