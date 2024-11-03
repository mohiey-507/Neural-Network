"""
optimizers.py

This module provides various optimization algorithms to update neural network weights during training.
Each optimizer uses a unique approach to adjusting learning rates and momentum for improved convergence.

Classes:
    - Optimizer: A base optimizer class providing a standard interface for all optimizers.
    - SGD: Stochastic Gradient Descent optimizer, performing weight updates proportional to gradients.
    - Momentum: Optimizer using momentum to accelerate gradient descent in optimal directions.
    - Adam: Adaptive moment estimation optimizer, utilizing adaptive learning rates for each parameter.

Usage:
    Each optimizer class implements the `setup` method to initialize any necessary state, and the `update`
    method to adjust weights according to the gradient for each layer parameter.

Example:
    >>> from optimizers import SGD, Momentum, Adam
    >>> sgd_optimizer = SGD(learning_rate=0.01)
    >>> momentum_optimizer = Momentum(learning_rate=0.01, momentum=0.9)
    >>> adam_optimizer = Adam(learning_rate=0.001)

Dependencies:
    - numpy
    - typing.List
    - neural_network.OptimizerProtocol
    - neural_network.LayerProtocol
"""

import numpy as np
from typing import List
from neural_network.base import OptimizerProtocol, LayerProtocol

class Optimizer(OptimizerProtocol):
    """
    Base optimizer class providing a common interface for weight update strategies.
    
    Attributes:
        learning_rate (float): Base learning rate for weight updates.
        layers (List[LayerProtocol]): Layers to be optimized.
    
    Methods:
        setup(layers): Initialize optimizer state for all layers.
        update(layer, gradient_key, gradient): Update layer parameters.
    """
    def __init__(
            self, 
            learning_rate: float
    ):
        self.learning_rate = learning_rate
        self.layers = None

    def setup(
            self, 
            layers: List[LayerProtocol]
    ) -> None:
        """
        Initializes optimizer state for each layer.

        Args:
            layers (List[LayerProtocol]): A list of layers that the optimizer will update. 
            Each layer should implement the LayerProtocol interface, providing access to parameters 
            (weights, biases, etc.) that need to be optimized.

        This method is meant to be overridden by subclasses to initialize any necessary state 
        specific to the optimizer being implemented. It sets the internal `layers` attribute 
        to the provided list of layers.
        """
        self.layers = layers

    def update(
            self, 
            layer: LayerProtocol, 
            gradient_key: str, 
            gradient: np.ndarray
    ) -> None:
        """
        Abstract method to update layer parameters using the gradient.

        Args:
            layer (LayerProtocol): The layer whose parameters are to be updated. 
            It should have attributes corresponding to the `gradient_key`.
            gradient_key (str): The key that specifies which parameter of the layer to update 
            (e.g., 'weights', 'bias').
            gradient (np.ndarray): The computed gradient with respect to the specified parameter. 

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
    

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    Performs simple weight updates proportional to the learning rate and gradient.
    
    Args:
        learning_rate (float): Step size for weight updates.
    """
    def __init__(
            self, 
            learning_rate: float
    ):
        super().__init__(learning_rate)

    def update(
            self, 
            layer: LayerProtocol, 
            gradient_key: str, 
            gradient: np.ndarray
    ):
        """
        Performs parameter updates by directly subtracting the scaled gradient.

        Args:
            layer (LayerProtocol): The layer whose parameter is being updated.
            gradient_key (str): The key specifying the parameter to update (e.g., 'weights', 'bias').
            gradient (np.ndarray): The computed gradient for the specified parameter.

        This method retrieves the parameter from the specified layer using the provided `gradient_key` 
        and updates it by subtracting the product of the learning rate and the gradient. 
        It ensures that the layer's parameter is modified in place.
        """
        if hasattr(layer, gradient_key):
            parameter = getattr(layer, gradient_key)
            parameter -= self.learning_rate * gradient

class Momentum(Optimizer):
    """
    Momentum optimizer to accelerate SGD with velocity adjustments.
    
    Args:
        learning_rate (float): Step size for learning.
        momentum (float): Momentum factor, typically between 0 and 1.

    Methods:
        setup(layers): Initializes velocities for all layers.
        update(layer, gradient_key, gradient): Adjusts parameters using momentum for smoother convergence.
    """
    def __init__(
            self, 
            learning_rate: float = 0.01, 
            momentum: float = 0.9
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
    
    def setup(
            self, 
            layers: List[LayerProtocol]
    ) -> None:
        """
        Initializes velocity terms for each parameter in all layers.

        Args:
            layers (List[LayerProtocol]): A list of layers to be optimized.

        This method sets up the velocity terms for each layer parameter (weights and biases) 
        by initializing them to zero. It also calls the base class's `setup` method to 
        register the layers. The velocity terms are stored in a dictionary, allowing for 
        momentum-based updates during the optimization process.
        """
        super().setup(layers)
        self.velocities = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                self.velocities[f'layer_{i}_weights'] = np.zeros_like(layer.weights)
                if layer.use_bias:
                    self.velocities[f'layer_{i}_bias'] = np.zeros_like(layer.bias)
            elif hasattr(layer, 'gamma'):
                self.velocities[f'layer_{i}_gamma'] = np.zeros_like(layer.gamma)
                self.velocities[f'layer_{i}_beta'] = np.zeros_like(layer.beta)
    
    def update(
            self, 
            layer: LayerProtocol, 
            gradient_key: str, 
            gradient: np.ndarray
    ) -> None:
        """
        Updates layer parameters by applying momentum-enhanced velocity adjustments.

        Args:
            layer (LayerProtocol): The layer whose parameters are being updated.
            gradient_key (str): The key specifying which parameter to update (e.g., 'weights', 'bias').
            gradient (np.ndarray): The computed gradient for the specified parameter.

        This method calculates the velocity for the specified parameter using the momentum factor 
        and the gradient. It updates the parameter in place by subtracting the calculated velocity, 
        resulting in smoother convergence and reduced oscillations.
        """
        if hasattr(layer, gradient_key):
            velocity_key = f'layer_{self.layers.index(layer)}_{gradient_key}'
            self.velocities[velocity_key] = (self.momentum * self.velocities[velocity_key] + 
                                           self.learning_rate * gradient)
            parameter = getattr(layer, gradient_key)
            parameter -= self.velocities[velocity_key]

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer, combining RMSprop and momentum methods.
    
    Args:
        learning_rate (float): Base learning rate.
        beta1 (float): Decay rate for first moment estimates.
        beta2 (float): Decay rate for second moment estimates.
        epsilon (float): Small constant to prevent division by zero.

    Methods:
        setup(layers): Initializes first and second moment terms for each layer.
        update(layer, gradient_key, gradient): Updates parameters using bias-corrected moment estimates.
    """
    def __init__(
            self, 
            learning_rate: float = 0.001, 
            beta1: float = 0.9, 
            beta2: float = 0.999,
            epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def setup(
            self, 
            layers: List[LayerProtocol]
    ) -> None:
        """
        Initializes first and second moment vectors for all layer parameters.

        Args:
            layers (List[LayerProtocol]): A list of layers to be optimized.

        This method prepares the first and second moment estimates for each parameter in the 
        layers by initializing them to zero. It calls the base class's `setup` method to 
        register the layers. The first moment estimates are stored in `m`, while the second 
        moment estimates are stored in `v`, facilitating the adaptive learning rate calculations.
        """
        super().setup(layers)
        self.m = {}
        self.v = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                self.m[f'layer_{i}_weights'] = np.zeros_like(layer.weights)
                self.v[f'layer_{i}_weights'] = np.zeros_like(layer.weights)
                if layer.use_bias:
                    self.m[f'layer_{i}_bias'] = np.zeros_like(layer.bias)
                    self.v[f'layer_{i}_bias'] = np.zeros_like(layer.bias)
            elif hasattr(layer, 'gamma'):
                self.m[f'layer_{i}_gamma'] = np.zeros_like(layer.gamma)
                self.v[f'layer_{i}_gamma'] = np.zeros_like(layer.gamma)
                self.m[f'layer_{i}_beta'] = np.zeros_like(layer.beta)
                self.v[f'layer_{i}_beta'] = np.zeros_like(layer.beta)
    
    def update(
            self, 
            layer: LayerProtocol, 
            gradient_key: str, 
            gradient: np.ndarray
    ) -> None:
        """
        Updates layer parameters using adaptive moment estimation.

        Args:
            layer (LayerProtocol): The layer whose parameters are being updated.
            gradient_key (str): The key specifying which parameter to update (e.g., 'weights', 'bias').
            gradient (np.ndarray): The computed gradient for the specified parameter.

        This method calculates and applies updates to the layer parameters using adaptive 
        learning rates derived from the first and second moment estimates. It applies bias 
        correction to these estimates to improve stability and performance in training, 
        updating the parameter in place.
        """
        if hasattr(layer, gradient_key):
            self.t += 1
            param_key = f'layer_{self.layers.index(layer)}_{gradient_key}'
            
            # Update biased first moment estimate
            self.m[param_key] = (self.beta1 * self.m[param_key] + 
                               (1 - self.beta1) * gradient)
            
            # Update biased second raw moment estimate
            self.v[param_key] = (self.beta2 * self.v[param_key] + 
                               (1 - self.beta2) * np.square(gradient))
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_key] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_key] / (1 - self.beta2**self.t)
            
            # Update parameters
            parameter = getattr(layer, gradient_key)
            parameter -= (self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
