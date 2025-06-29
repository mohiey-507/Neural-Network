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
    def __init__(
            self, 
            name: Optional[str] = None
    ):
        super().__init__(name)

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
        # Store input shape for backward pass
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

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
        # Reshape gradient back to original input shape
        return gradient.reshape(self.input_shape)


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
            activation: Optional[str] = None,
            use_bias: bool = True,
            name: Optional[str] = None
    ):
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
        self.output_shape = (input_shape[0], self.units)
        fan_in, fan_out = input_dim, self.units
        if self.activation and self.activation.lower() in ('relu', 'leaky_relu', 'elu'):
            self.weights = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
        if self.use_bias:
            self.bias = np.zeros((1, self.units))
    
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
        out = gradient @ self.weights.T
        self.input = None
        self.output = None
        return out

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
        if not self.activation:
            return inputs

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
        if not self.activation:
            return gradient

        activation_gradient_fn = getattr(Activation, self.activation + '_gradient', None)
        if activation_gradient_fn is not None:
            return gradient * activation_gradient_fn(self.output)
        return gradient


class BatchNorm(BaseLayer):
    """
    Batch Normalization layer to normalize layer inputs.
    
    Normalizes input by subtracting mean and scaling by standard deviation. 
    
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
        self.gradients = {}
        
    def build(
        self, 
        input_shape: Tuple
    ) -> None:
        """
        Initializes BatchNorm layer parameters.

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
        if training:
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0) + self.epsilon

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Normalize
            x_centered = inputs - mean
            std_inv = 1.0 / np.sqrt(var)
            x_normalized = x_centered * std_inv

            # Cache only what is needed for backward
            self.cache = (x_centered, std_inv)
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.epsilon)
            x_centered = inputs - self.running_mean
            x_normalized = x_centered * std_inv
            self.cache = None
            
        out = self.gamma * x_normalized + self.beta
        return out
    
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
        x_centered, std_inv = self.cache
        N = gradient.shape[0]

        # Gradients for gamma and beta
        x_normalized = x_centered * std_inv
        self.gradients['gamma'] = np.sum(gradient * x_normalized, axis=0)
        self.gradients['beta'] = np.sum(gradient, axis=0)

        # Backprop through normalization
        dx_norm = gradient * self.gamma

        dvar = np.sum(dx_norm * x_centered * -0.5 * std_inv**3, axis=0)
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * x_centered, axis=0)

        dx = dx_norm * std_inv + dvar * 2.0 * x_centered / N + dmean / N

        self.cache = None
        return dx

class LayerNorm(BaseLayer):
    """Layer Normalization for 3-D tensors (B, S, D)"""
    def __init__(
        self, 
        epsilon: float = 1e-5, 
        name: Optional[str] = None
    ):
        super().__init__(name)
        self.epsilon = epsilon
        self.gamma: np.ndarray = None  # (1, 1, D)
        self.beta: np.ndarray = None   # (1, 1, D)
        self.cache = None
        self.gradients = {}

    def build(
        self, 
        input_shape: Tuple
    ):
        super().build(input_shape)
        D = input_shape[-1]
        self.gamma = np.ones((1, 1, D))
        self.beta = np.zeros((1, 1, D))
        self.output_shape = input_shape

    def forward(
        self, 
        inputs: np.ndarray, 
        training: bool = True
    ) -> np.ndarray:
        mean = inputs.mean(axis=-1, keepdims=True)
        var = inputs.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        x_norm = (inputs - mean) * std_inv
        self.cache = (x_norm, std_inv)
        return self.gamma * x_norm + self.beta

    def backward(
        self, 
        gradient: np.ndarray
    ) -> np.ndarray:
        x_norm, std_inv = self.cache
        self.gradients['gamma'] = np.sum(gradient * x_norm, axis=(0, 1), keepdims=True)
        self.gradients['beta'] = np.sum(gradient, axis=(0, 1), keepdims=True)

        N = x_norm.shape[-1]  # feature dimension (D)
        dx_norm = gradient * self.gamma

        # Backprop through normalization
        sum_dx = np.sum(dx_norm, axis=-1, keepdims=True)
        sum_dx_xnorm = np.sum(dx_norm * x_norm, axis=-1, keepdims=True)

        dx = (dx_norm - sum_dx / N - x_norm * sum_dx_xnorm / N) * std_inv

        self.cache = None
        return dx


class Embedding(BaseLayer):
    """Embedding layer mapping integer indices (B, S) to dense vectors (B, S, D)."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_scale: float = 0.01,
        mask_zero: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_scale = init_scale
        self.mask_zero = mask_zero
        self.embeddings: np.ndarray = None  # (V, D)
        self.gradients = {}


    def build(
        self, 
        input_shape: Tuple
    ) -> None:
        """
        Initializes the embedding layer with random weights.

        Args:
            input_shape (Tuple): Shape of the input data, including batch size,
            expected as (B, S).
        """
        super().build(input_shape) # (B, S)
        self.embeddings = np.random.uniform(-self.init_scale, self.init_scale, size=(self.vocab_size, self.embed_dim)).astype(float)
        self.output_shape = (input_shape[0], input_shape[1], self.embed_dim)

    def forward(
        self, 
        inputs: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        self.input = inputs.astype(int)
        out = self.embeddings[self.input]
        return out

    def backward(
        self, 
        gradient: np.ndarray
    ) -> np.ndarray:
        # gradient shape: (B, S, D)
        self.gradients['embeddings'] = np.zeros_like(self.embeddings)
        indices = self.input.flatten()
        grad_flat = gradient.reshape(-1, self.embed_dim)

        # If padding index 0 should be ignored
        if self.mask_zero:
            mask = indices != 0
            np.add.at(self.gradients['embeddings'], indices[mask], grad_flat[mask])
        else:
            np.add.at(self.gradients['embeddings'], indices, grad_flat)

        self.input = None
        # No gradient flows to the integer inputs
        return np.zeros((gradient.shape[0], gradient.shape[1]))


class MultiHeadAttention(BaseLayer):
    """Multi-Head Self-Attention working completely in (B, S, D) space.

    Internally uses two shared `Dense` sub-layers:
        • `qkv_dense`: single linear that projects input to concatenated Q, K, V.
        • `out_dense`: linear that projects concatenated context back to model dim.

    This keeps weight initialisation / optimiser logic in one place (the `Dense`
    implementation) and guarantees shape compatibility with the rest of the
    framework (Dense expects 2-D (B, F) input, so we temporarily flatten the
    (B, S, D) sequence to (B·S, D)).
    """
    def __init__(self, embed_dim: int, num_heads: int, name: Optional[str] = None):
        super().__init__(name)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Sub-layers (exposed for optimizers)
        self.qkv_dense = Dense(units=3 * embed_dim, use_bias=False)
        self.out_dense = Dense(units=embed_dim, use_bias=False)
        self.sub_layers = [self.qkv_dense, self.out_dense]

        # caches / grads
        self.attn_weights = None
        self.projections = None  # Qh, Kh, Vh
        self.gradients = {}

    # ---------------------------------------------------------------------
    # helpers
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, S, D) → (B, H, S, D/H)"""
        B, S, _ = x.shape
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, H, S, D/H) → (B, S, D)"""
        B, H, S, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)

    # ---------------------------------------------------------------------
    def build(self, input_shape: Tuple):
        """Build sub dense layers based on (B, S, D) input shape."""
        super().build(input_shape)
        B, S, D = input_shape
        assert D == self.embed_dim, "Input last dim must equal embed_dim"

        flat_shape = (B * S, D)
        self.qkv_dense.build(flat_shape)
        self.out_dense.build(flat_shape)
        self.output_shape = input_shape

    # ---------------------------------------------------------------------
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs  # cache for residual math
        B, S, D = inputs.shape

        # 1. QKV projection in a single matmul via Dense
        flat_in = inputs.reshape(B * S, D)
        qkv = self.qkv_dense.forward(flat_in, training)  # (B*S, 3D)
        qkv = qkv.reshape(B, S, 3 * D)
        Q, K, V = np.split(qkv, 3, axis=-1)  # each (B,S,D)

        # 2. split into heads
        Qh, Kh, Vh = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        # 3. scaled dot-product attention
        scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)  # (B,H,S,S)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)

        context = weights @ Vh  # (B,H,S,D/H)
        context_combined = self._combine_heads(context)  # (B,S,D)

        # 4. output projection
        out_flat = context_combined.reshape(B * S, D)
        out_proj = self.out_dense.forward(out_flat, training)
        self.output = out_proj.reshape(B, S, D)

        # cache for backward
        self.attn_weights = weights
        self.projections = (Qh, Kh, Vh)
        self.context = context_combined
        return self.output

    # ---------------------------------------------------------------------
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        B, S, D = gradient.shape

        # 1. Backprop through output Dense
        grad_flat = gradient.reshape(B * S, D)
        d_context_combined_flat = self.out_dense.backward(grad_flat)  # (B*S, D)
        d_context_combined = d_context_combined_flat.reshape(B, S, D)

        # 2. Backprop through attention combine
        d_context_h = self._split_heads(d_context_combined)  # (B,H,S,D/H)

        Qh, Kh, Vh = self.projections
        weights = self.attn_weights

        # Grad w.r.t attention weights and V
        d_weights = d_context_h @ Vh.transpose(0, 1, 3, 2)  # (B,H,S,S)
        d_Vh = weights.transpose(0, 1, 3, 2) @ d_context_h  # (B,H,S,D/H)

        # softmax derivative
        dw_times_w = d_weights * weights
        d_scores = dw_times_w - weights * dw_times_w.sum(axis=-1, keepdims=True)
        d_scores /= np.sqrt(self.head_dim)

        d_Qh = d_scores @ Kh
        d_Kh = d_scores.transpose(0, 1, 3, 2) @ Qh

        # merge head grads
        d_Q = self._combine_heads(d_Qh)  # (B,S,D)
        d_K = self._combine_heads(d_Kh)
        d_V = self._combine_heads(d_Vh)

        # 3. Concatenate Q,K,V grads and send through QKV dense
        d_qkv = np.concatenate([d_Q, d_K, d_V], axis=-1)  # (B,S,3D)
        d_qkv_flat = d_qkv.reshape(B * S, 3 * D)
        d_inputs_flat = self.qkv_dense.backward(d_qkv_flat)  # (B*S, D)
        d_inputs = d_inputs_flat.reshape(B, S, D)

        return d_inputs


class TransformerEncoderLayer(BaseLayer):
    """
    Transformer Encoder layer implementing the standard Transformer architecture.
    
    Architecture:
    Input (B,S,D) → LayerNorm → MHA → Add & Norm → FFN → Add & Norm → Output (B,S,D)
    
    Args:
        embed_dim (int): Dimension of the embedding space (D).
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward network's inner layer.
        ff_activation (str): Activation for the FFN. Defaults to 'relu'.
        dropout_rate (float): Dropout rate for regularization.
        name (Optional[str]): Name of the layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        ff_activation: str = 'relu',
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Attention sub-layer
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network sub-layers
        self.ffn1 = Dense(ff_dim, activation=ff_activation)
        self.ffn2 = Dense(embed_dim)
                
        # Normalization layers
        self.ln1 = LayerNorm()
        self.ln2 = LayerNorm()
        
        # Dropout layers
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
        self.sub_layers = [self.attn, self.ffn1, self.ffn2, self.ln1, self.ln2, self.dropout1, self.dropout2]
        self.gradients = {}

    def build(self, input_shape: Tuple):
        """Build sub-layers based on input shape."""
        super().build(input_shape)
        B, S, D = input_shape
        assert D == self.embed_dim, f"Input dimension {D} does not match embed_dim {self.embed_dim}"
        
        # Build sub-layers
        ffn_input_shape_flat = (B * S, D)
        self.ffn1.build(ffn_input_shape_flat)
        self.ffn2.build((B * S, self.ff_dim))

        self.ln1.build(input_shape)
        self.attn.build(input_shape)
        self.ln2.build(input_shape)
        
        self.output_shape = input_shape

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the Transformer Encoder layer.
        
        Args:
            inputs: Input tensor of shape (B, S, D)
            training: Whether to run in training mode (affects dropout)
            
        Returns:
            Output tensor of shape (B, S, D)
        """
        self.input = inputs
        
        # --- First Sub-layer: Multi-Head Attention ---
        # Pre-normalization
        ln1_out = self.ln1.forward(inputs, training)

        # Attention
        attn_output = self.attn.forward(ln1_out, training)

        # Dropout and residual connection
        attn_res = inputs + self.dropout1.forward(attn_output, training)
        
        # --- Second Sub-layer: Feed-Forward Network ---
        # Pre-normalization
        ln2_out = self.ln2.forward(attn_res, training)
        
        # FFN processing
        B, S, D = ln2_out.shape
        ffn_in_flat = ln2_out.reshape(B * S, D)
        ffn_hidden = self.ffn1.forward(ffn_in_flat, training)
        ffn_output_flat = self.ffn2.forward(ffn_hidden, training)
        ffn_output = ffn_output_flat.reshape(B, S, D)
        
        # Dropout and residual connection
        self.output = attn_res + self.dropout2.forward(ffn_output, training)
        
        return self.output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through the Transformer Encoder layer.
        
        Args:
            gradient: Gradient tensor of shape (B, S, D) from the next layer.
            
        Returns:
            Gradient with respect to the layer's inputs.
        """
        # --- Backprop through the second sub-layer (FFN) ---
        # Backprop through the final residual connection
        grad_attn_res = gradient.copy() # Gradient for the output of the first sub-layer
        grad_ffn_dropout = gradient.copy() # Gradient for the output of the dropout layer

        # Backprop through dropout2
        grad_ffn_output = self.dropout2.backward(grad_ffn_dropout)
        
        # Reshape for FFN backprop
        B, S, D = grad_ffn_output.shape
        grad_ffn_output_flat = grad_ffn_output.reshape(B * S, D)
        
        # Backprop through ffn2 and ffn1
        grad_ffn_hidden = self.ffn2.backward(grad_ffn_output_flat)
        grad_ffn_in_flat = self.ffn1.backward(grad_ffn_hidden)
        
        # Reshape back to 3D
        grad_ln2_out = grad_ffn_in_flat.reshape(B, S, D)
        grad_attn_res += self.ln2.backward(grad_ln2_out)
        
        # --- Backprop through the first sub-layer (MHA) ---
        # Backprop through the first residual connection
        grad_inputs = grad_attn_res.copy() # Gradient for the original inputs
        grad_attn_dropout = grad_attn_res.copy() # Gradient for the dropout1 output
        
        grad_attn_output = self.dropout1.backward(grad_attn_dropout)
        grad_ln1_out = self.attn.backward(grad_attn_output)
        grad_inputs += self.ln1.backward(grad_ln1_out)
        
        return grad_inputs
