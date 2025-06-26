import numpy as np
from typing import Optional, Tuple
from neural_network.base import BaseLayer
from neural_network.activation import Activation

class Flatten(BaseLayer):
    """
    Flatten layer transforms multidimensional input tensors into 2D tensors.
    
    In the forward pass, it reshapes the input to (batch_size, flattened_features).
    In the backward pass, it restores the original input shape.
    
    Attributes:
        input (np.ndarray): Original input tensor before flattening.
        output (np.ndarray): Flattened output tensor.
    """
    def forward(
            self,
            inputs: np.ndarray, 
            training: bool = True
    ) -> np.ndarray:
        """
        Forward pass to reshape the input tensor.

        Args:
            inputs (np.ndarray): Input tensor to flatten.
            training (bool): Indicates whether the layer is in training mode. Defaults to True.
        
        Returns:
            np.ndarray: Flattened output tensor.
        """
        self.input = inputs
        self.output = inputs.reshape(inputs.shape[0], -1)
        return self.output

    def backward(
            self, 
            gradient: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass to reshape the gradient tensor back to the original shape.

        Args:
            gradient (np.ndarray): Gradient tensor from the subsequent layer.
        
        Returns:
            np.ndarray: Reshaped gradient tensor with the same shape as the input.
        """
        return gradient.reshape(self.input.shape)


class Dropout(BaseLayer):
    """
    Dropout regularization layer to reduce overfitting.
    
    During training, randomly sets a fraction of input units to zero based on the 
    specified dropout rate. During inference, all units are kept with their original values.
    
    Attributes:
        rate (float): Fraction of units to drop during training.
        mask (np.ndarray): Binary mask used to zero out units during training.
    
    Args:
        rate (float): Dropout rate between 0 and 1.
        name (Optional[str]): Optional name for the layer.
    """
    def __init__(
            self, 
            rate: float, 
            name: Optional[str] = None
    ):
        """
        Initializes the Dropout layer.

        Args:
            rate (float): Fraction of input units to drop (0 < rate < 1).
            name (Optional[str]): Optional name for the layer.
        """
        super().__init__(name)
        self.rate = rate
        self.mask = None

    def forward(
            self, 
            inputs:np.ndarray, 
            training: bool = True
    ) -> np.ndarray:
        """
        Forward pass applying dropout during training.

        Args:
            inputs (np.ndarray): Input tensor.
            training (bool): Indicates whether the layer is in training mode. Defaults to True.
        
        Returns:
            np.ndarray: Output tensor after applying dropout.
        """
        self.input = inputs
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate) # rescales the mask
            return inputs * self.mask
        return inputs

    def backward(
            self, 
            gradient: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass scaling the gradient by the dropout mask.

        Args:
            gradient (np.ndarray): Gradient tensor from the subsequent layer.
        
        Returns:
            np.ndarray: Gradient tensor with dropout mask applied.
        """
        return gradient * self.mask


class Dense(BaseLayer):
    """
    Fully connected (dense) neural network layer.
    
    Implements linear transformation: output = activation(dot(input, weights) + bias)
    Supports various activation functions and optional bias.
    
    Attributes:
        units (int): Number of neurons in the layer.
        activation (Optional[str]): Name of the activation function to apply.
        weights (np.ndarray): Layer weight matrix.
        bias (Optional[np.ndarray]): Layer bias vector.
        gradients (dict): Computed gradients for weights and bias.
    
    Args:
        units (int): Number of neurons in the layer.
        activation (Optional[str]): Name of the activation function.
        use_bias (bool): Whether to include a bias term. Defaults to True.
        name (Optional[str]): Optional name for the layer.
    """
    def __init__(
            self, units: int,
            activation: str = Optional[str],
            use_bias: bool = True,
            name: Optional[str] = None
    ):
        """
        Initializes the Dense layer.

        Args:
            units (int): Number of neurons in the layer.
            activation (Optional[str]): Name of the activation function.
            use_bias (bool): Whether to include a bias term. Defaults to True.
            name (Optional[str]): Optional name for the layer.
        """
        super().__init__(name)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.gradients = {}

    def build(
            self, 
            input_shape: Tuple
    ) -> None:
        """
        Initializes weights and biases for the layer.

        Args:
            input_shape (Tuple): Shape of the input tensor.
        """
        super().build(input_shape)
        input_dim = np.prod(input_shape[1:])
        self.weights = np.random.randn(input_dim, self.units) * 0.01
        self.bias = np.zeros((1, self.units)) if self.use_bias else None
        self.output_shape = (input_shape[0], self.units)

    def forward(
            self, 
            inputs: np.ndarray, 
            training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through the dense layer.

        Args:
            inputs (np.ndarray): Input tensor.
            training (bool): Indicates whether the layer is in training mode. Defaults to True.
        
        Returns:
            np.ndarray: Output tensor after applying linear transformation and activation.
        """
        self.input = inputs
        output = np.dot(inputs, self.weights)
        if self.use_bias:
            output += self.bias
        self.output = self._apply_activation(output)
        return self.output
    
    def backward(self,
                gradient: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass to compute gradients with respect to weights, bias, and inputs.

        Args:
            gradient (np.ndarray): Gradient tensor from the subsequent layer.
        
        Returns:
            np.ndarray: Gradient tensor with respect to the layer's input.
        """
        gradient = self._apply_activation_gradient(gradient)
        self.gradients['weights'] = np.dot(self.input.T, gradient)
        if self.use_bias:
            self.gradients['bias'] = np.sum(gradient, axis=0, keepdims=True)
        return np.dot(gradient, self.weights.T)

    def _apply_activation(
            self, 
            inputs: np.ndarray
    ) -> np.ndarray:
        """
        Applies the specified activation function to the input.

        Args:
            inputs (np.ndarray): Input tensor to apply activation.
        
        Returns:
            np.ndarray: Output tensor after applying the activation function.
        """
        activation_fn = getattr(Activation, self.activation, None)

        if activation_fn is not None:
            return activation_fn(inputs)
        return inputs

    def _apply_activation_gradient(
            self, 
            gradient: np.ndarray
    ) -> np.ndarray:
        """
        Applies the gradient of the activation function to the gradient tensor.

        Args:
            gradient (np.ndarray): Gradient tensor from the subsequent layer.
        
        Returns:
            np.ndarray: Gradient tensor after applying the activation function gradient.
        """
        activation_gradient_fn = getattr(Activation, self.activation + '_gradient', None)

        if activation_gradient_fn is not None:
            return gradient * activation_gradient_fn(self.output)
        return gradient


class BatchNorm(BaseLayer):
    """
    Batch Normalization layer to normalize layer inputs.
    
    Normalizes input by subtracting mean and scaling by standard deviation. 
    Maintains running statistics during training for use during inference.
    
    Attributes:
        epsilon (float): Small value to prevent division by zero.
        momentum (float): Momentum for updating running statistics.
        gamma (np.ndarray): Learnable scaling parameter.
        beta (np.ndarray): Learnable shifting parameter.
        running_mean (np.ndarray): Exponential moving average of batch means.
        running_var (np.ndarray): Exponential moving average of batch variances.
    
    Args:
        epsilon (float): Small value to prevent division by zero. Defaults to 1e-7.
        momentum (float): Momentum for updating running statistics. Defaults to 0.9.
        name (Optional[str]): Optional name for the layer.
    """ 
    def __init__(
            self, 
            epsilon: float = 1e-7, 
            momentum: float = 0.9, 
            name: Optional[str] = None):
        """
        Initializes the BatchNorm layer with the specified epsilon and momentum.

        Args:
            epsilon (float): Small constant to prevent division by zero.
            momentum (float): Momentum for updating running statistics.
            name (Optional[str]): Name of the layer, if provided.
        """
        super().__init__(name)
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.cache = None
        self.gradients = {}  # Initialize gradients dictionary
        
    def build(
        self, 
        input_shape: Tuple
    ) -> None:
        """
        Builds the BatchNorm layer by initializing parameters based on input shape.

        Args:
            input_shape (Tuple): Shape of the input data, including batch size.
        """
        super().build(input_shape)
        self.output_shape = input_shape
        self.gamma = np.ones(input_shape[1:])
        self.beta = np.zeros(input_shape[1:])
        self.running_mean = np.zeros(input_shape[1:])
        self.running_var = np.ones(input_shape[1:])
        
    def forward(
            self, 
            inputs: np.ndarray,
            training: bool = True
    ) -> np.ndarray:
        """
        Performs the forward pass of the BatchNorm layer.

        During training, it calculates the mean and variance of the inputs, updates the
        running mean and variance, and normalizes the inputs. During inference, it 
        uses the running statistics to normalize the inputs.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, num_features).
            training (bool): Indicates whether the model is in training mode. 
                            Defaults to True.

        Returns:
            np.ndarray: Normalized output of the layer.
        """
        self.input = inputs
        
        if training:
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0) + self.epsilon
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalize
            x_normalized = (inputs - mean) / np.sqrt(var)
            self.cache = (x_normalized, mean, var, inputs)
        else:
            x_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            
        self.output = self.gamma * x_normalized + self.beta
        return self.output
    
    def backward(
            self, 
            gradient: np.ndarray
    ) -> np.ndarray:
        """
        Performs the backward pass of the BatchNorm layer.

        Computes gradients with respect to the input, gamma, and beta parameters.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the output of 
                                    the BatchNorm layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the layer.
        """
        x_normalized, mean, var, x = self.cache
        N = gradient.shape[0]
        
        # Compute gradients for gamma and beta
        self.gradients['gamma'] = np.sum(gradient * x_normalized, axis=0)
        self.gradients['beta'] = np.sum(gradient, axis=0)
        
        # Compute gradient with respect to normalized input
        dx_normalized = gradient * self.gamma
        
        # Compute gradients with respect to input
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * var**(-1.5), axis=0)
        dmean = np.sum(dx_normalized * -1/np.sqrt(var), axis=0) + dvar * np.mean(-2.0 * (x - mean), axis=0)
        
        dx = dx_normalized / np.sqrt(var) 
        dx += 2.0 * dvar * (x - mean) / N
        dx += dmean / N
        
        return dx