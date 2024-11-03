# Neural Network Implementation

A lightweight neural network library implemented in Python, providing core functionality for building, training, and evaluating neural networks. This implementation includes essential layer types, optimization algorithms, and training utilities.

## Features

- **Modular Architecture**: Build neural networks by stacking layers
- **Multiple Layer Types**:
  - Dense (Fully Connected)
  - Batch Normalization
  - Dropout
  - Flatten
- **Training Features**:
  - Mini-batch training
  - Multiple optimization algorithms (SGD, Momentum, Adam)
  - Training history tracking
  - Validation during training
- **Built-in Regularization**:
  - Dropout layers
  - Batch normalization
- **Flexible Model Configuration**:
  - Configurable loss functions
  - Multiple activation functions
  - Customizable optimization parameters

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/neural-network-implementation.git
cd neural-network-implementation
```

Install dependencies:
```bash
pip install numpy
```

## Usage

### Basic Example

```python
from neural_network import Network
from neural_network.layers import Dense, BatchNorm, Dropout

# Create model
model = Network()

# Add layers
model.add(Dense(32, activation='relu'))
model.add(BatchNorm())
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    loss='binary_cross_entropy',
    optimizer='adam',
    optimizer_params={'learning_rate': 0.001}
)

# Train model
history = model.fit(
    X_train, 
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

## Project Structure

- `network.py`: Core Network class implementation
- `layers.py`: Implementation of various layer types
- `main.py`: Example usage and test cases
- Supporting modules:
  - Activation functions
  - Loss functions
  - Optimizers
  - Base classes and protocols

## Available Layers

### Dense Layer
```python
Dense(units, activation=None, use_bias=True)
```
- Fully connected layer with optional activation function
- Configurable number of units and bias

### Batch Normalization
```python
BatchNorm(epsilon=1e-7, momentum=0.9)
```
- Normalizes layer inputs
- Reduces internal covariate shift
- Configurable epsilon and momentum

### Dropout
```python
Dropout(rate)
```
- Applies dropout regularization
- Helps prevent overfitting
- Configurable dropout rate

### Flatten
```python
Flatten()
```
- Reshapes input to 2D tensor
- Used for transitioning between convolutional and dense layers

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Inspired by modern deep learning frameworks
- Built for educational purposes and practical applications
- Demonstrates core concepts of neural network implementation

