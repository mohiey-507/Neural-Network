"""
Neural Network Library Initialization

This library provides a flexible, modular implementation of a neural network from scratch using NumPy.

Modules:
- base.py: Defines base classes and protocols for layers and optimizers.
- layers.py: Implements various neural network layer types including Dense, Flatten, Dropout, and BatchNorm.
- network.py: Provides the main Network class for model creation and training.
- optimizers.py: Implements different optimization algorithms such as SGD, Momentum, and Adam.
- losses.py: (Assumed to exist) Contains loss functions for model training.

Key Components:
- Layers: Dense, Flatten, Dropout, BatchNorm
- Optimizers: SGD, Momentum, Adam
- Supported Activation Functions: Implemented in a separate activation module

Usage:
To use this library, import the necessary classes and create a neural network by stacking layers and specifying loss functions and optimizers. Train the model on your data and use it for predictions.

Example:
```python
from neural_network import Network, Dense, Flatten, Dropout, SGD, Loss

# Create a neural network
model = Network()

# Add layers to the model
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # For a classification task

# Compile the model with a loss function and optimizer
model.compile(loss='categorical_cross_entropy', optimizer=SGD(lr=0.01))

# Fit the model on training data
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Make predictions on new data
predictions = model.predict(X_test)
"""

from .base import BaseLayer, LayerProtocol, OptimizerProtocol
from .layers import Dense, Flatten, Dropout, BatchNorm
from .network import Network
from .optimizers import SGD, Momentum, Adam
from .activation import Activation

__all__ = [
    # Base classes and protocols
    'BaseLayer', 'LayerProtocol', 'OptimizerProtocol',
    
    # Layer types
    'Dense', 'Flatten', 'Dropout', 'BatchNorm',
    
    # Network and Optimizers
    'Network', 'SGD', 'Momentum', 'Adam',

    # Activation functions
    'Activation'
]

__version__ = '0.1.0'
