# Neural-Network: Build Custom Neural Nets with Pure NumPy

A lightweight, modular neural network library in pure NumPy—stack layers, perfect for learning or prototyping deep learning from scratch.

## Features

- **Modular Layer Stacking**: Build networks by adding layers like Dense, BatchNorm, Dropout, and more.
- **Transformer Support**: Includes MultiHeadAttention, PositionalEncoding, and more for sequence models.
- **Optimizers & Losses**: SGD, Momentum, Adam; MSE, Binary/Categorical Cross-Entropy.
- **Training & Inference**: Fit models with batch support, evaluate accuracy, and predict seamlessly.
- **No Frameworks Needed**: Runs on NumPy alone for ultimate control and transparency.

## Installation

Clone the repo and install as a package:

```bash
git clone https://github.com/mohiey-507/Neural-Network.git
cd Neural-Network
pip install -e .
```

## Quick Start

Import and build a simple network:

```python
import neural_network as nn

# Build model
model = nn.Network()
model.add(nn.Dense(units=32, activation='relu'))
model.add(nn.Dense(units=1, activation='sigmoid'))

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

For a full example on IMDB sentiment analysis with transformers, check this [Kaggle Notebook](https://www.kaggle.com/code/mohamedmohiey/numpy-transformer).

## Available Layers

### Flatten

```python
Flatten()
```

- Reshapes multi-D input to (batch_size, flattened_features).
- Input: (B, ...) → Output: (B, flattened).

### Dropout

```python
Dropout(rate=0.5)
```

- Randomly zeros units during training, scales by 1/(1-rate).
- Input: Any shape → Output: Same shape.

### Dense

```python
Dense(units=64, activation='relu', use_bias=True)
```

- Linear transform + optional activation.
- Input: (B, in_features) → Output: (B, units).

### BatchNorm

```python
BatchNorm(epsilon=1e-7, momentum=0.9)
```

- Normalizes across batch, updates running stats during training.
- Input: (B, features) → Output: (B, features).

### LayerNorm

```python
LayerNorm(epsilon=1e-5)
```

- Forward: Normalizes across last dimension.
- Backward: Propagates through per-feature mean/var.
- Input: (B, S, D) → Output: (B, S, D).

### Embedding

```python
Embedding(vocab_size=10000, embed_dim=128, init_scale=0.01, mask_zero=False)
```

- Forward: Maps integer indices to dense vectors, optional zero-masking.
- Backward: Accumulates grads to embedding matrix, ignores input grad.
- Input: (B, S) → Output: (B, S, D) + optional mask.

### MultiHeadAttention

```python
MultiHeadAttention(embed_dim=256, num_heads=8)
```

- Forward: Scaled dot-product attention across heads, projects back.
- Backward: Propagates through projections and softmax.
- Input: (B, S, D) → Output: (B, S, D).

### TransformerEncoderLayer

```python
TransformerEncoderLayer(embed_dim=256, num_heads=8, ff_dim=512, ff_activation='relu', dropout_rate=0.1)
```

- Forward: LayerNorm → MHA → Add/Norm → FFN → Add/Norm.
- Backward: Propagates through sub-layers in reverse.
- Input: (B, S, D) → Output: (B, S, D).

### PositionalEncoding

```python
PositionalEncoding(max_len=512, embed_dim=256)
```

- Forward: Adds fixed sine/cosine encodings.
- Backward: Passes gradient unchanged.
- Input: (B, S, D) → Output: (B, S, D).

### GlobalAveragePooling1D

```python
GlobalAveragePooling1D()
```

- Forward: Averages across sequence, handles masking.
- Backward: Distributes gradient to non-padded positions.
- Input: (B, S, D) → Output: (B, D).

## Optimizers

- `SGD(learning_rate=0.01)`: Basic gradient descent.
- `Momentum(learning_rate=0.01, momentum=0.9)`: Adds velocity for faster convergence.
- `Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`: Adaptive rates with momentum.

## Losses

- `mean_squared_error`: Regression tasks.
- `binary_cross_entropy`: Binary classification.
- `categorical_cross_entropy`: Multi-class classification.

## Contributing

Fork the repo, add features or fixes, and submit a PR. Focus on NumPy-only enhancements.
