"""
base.py

This module provides the foundational protocols and base class definitions for building 
a neural network framework. It includes protocols for layer and optimizer interfaces, 
and an abstract base class `BaseLayer` which defines common properties and methods 
for all neural network layers.

Classes:
    - LayerProtocol: Protocol for defining the interface of layers, with methods for 
        forward and backward passes and a build function for initialization.
    - OptimizerProtocol: Protocol for optimizers, requiring setup and update methods 
        to handle parameter updates for each layer.
    - BaseLayer: Abstract base class for all neural network layers, providing standard 
        attributes and methods such as forward and backward propagation, and a build method 
        to initialize shapes.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Protocol, List, Optional

class LayerProtocol(Protocol):
    """
    Protocol defining the interface for layers.

    Attributes:
        name (str): The name of the layer.
        trainable (bool): Whether the layer is trainable.
        input_shape (Optional[Tuple[int, ...]]): Shape of the input data.
        output_shape (Optional[Tuple[int, ...]]): Shape of the output data.
    
    Methods:
        forward(inputs: np.ndarray, training: bool = True) -> np.ndarray:
            Executes the forward pass and returns the layer's output.
        
        backward(gradient: np.ndarray) -> np.ndarray:
            Executes the backward pass and returns the computed gradient.
        
        build(input_shape: Tuple[int, ...]) -> None:
            Initializes the layer with the specified input shape.
    """
    name: str
    trainable: bool
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray: ...
    def backward(self, gradient: np.ndarray) -> np.ndarray: ...
    def build(self, input_shape: Tuple[int, ...]) -> None: ...

class OptimizerProtocol(Protocol):
    """
    Protocol defining the interface for optimizers.

    Methods:
        setup(layers: List[LayerProtocol]) -> None:
            Configures the optimizer with the specified layers.
        
        update(layer: LayerProtocol, gradient_key: str, gradient: np.ndarray) -> None:
            Updates the layer parameters based on the computed gradient.
    """
    def setup(self, layers: List['LayerProtocol']) -> None: ...
    def update(self, layer: 'LayerProtocol', gradient_key: str, gradient: np.ndarray) -> None: ...


class BaseLayer(ABC):
    """
    Abstract base class for all layers in the neural network, defining core attributes 
    and methods for forward and backward propagation.

    Attributes:
        name (str): Name of the layer, defaults to the class name if not provided.
        trainable (bool): Whether the layer is trainable. Defaults to True.
        input_shape (Optional[Tuple[int, ...]]): Shape of the input data.
        input (Optional[np.ndarray]): The input data for the layer.
        output_shape (Optional[Tuple[int, ...]]): Shape of the output data.
        output (Optional[np.ndarray]): The output data of the layer.
    
    Methods:
        forward(input: np.ndarray, training: bool = True) -> np.ndarray:
            Abstract method for forward pass, to be implemented by subclasses.
        
        backward(gradient: np.ndarray) -> np.ndarray:
            Abstract method for backward pass, to be implemented by subclasses.
        
        build(input_shape: Tuple[int, ...]) -> None:
            Sets the input and output shapes for the layer based on the provided input shape.
    """
    def __init__(
            self, 
            name: str = None
    ):
        """
        Initializes the layer with a specified name or defaults to the class name.
        
        Args:
            name (str, optional): Name of the layer.
        """
        self.name = name if name else self.__class__.__name__
        self.trainable = True
        self.input_shape = None
        self.input = None
        self.output_shape = None
        self.output = None

    @abstractmethod
    def forward(
        self, 
        inputs: np.ndarray, 
        training: bool = True
    ) -> np.ndarray:
        """
        Abstract method to perform the forward pass. Must be implemented by subclasses.

        Args:
            input (np.ndarray): Input data.
            training (bool): Whether the layer is in training mode.

        Returns:
            np.ndarray: Output after forward pass.
        """
        pass

    @abstractmethod
    def backward(
        self, 
        gradient: np.ndarray
    ) -> np.ndarray:
        """
        Abstract method to perform the backward pass. Must be implemented by subclasses.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input.
        """
        pass

    def build(
            self, 
            input_shape: Tuple[int, ...]
    ) -> None:
        """
        Initializes the layer's input and output shapes based on the specified input shape.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input data.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape
