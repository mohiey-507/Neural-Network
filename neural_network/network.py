"""
network.py

This module provides the core Network class for building and training neural networks.
Supports layer composition, loss function and optimizer configuration, and model training.

Classes:
    - Network: Main neural network class for building, compiling, training, and 
        making predictions with a stack of layers.

Key Features:
    - Dynamic layer addition
    - Configurable loss functions and optimizers
    - Batch training with optional validation
    - Performance tracking via training history

Usage:
    To use the Network class, create an instance of the class, add layers, compile the 
    model with a loss function and optimizer, and then call the fit method to train the 
    network. For example:

        from network import Network
        from layers import Dense, Dropout
        
        # Create a network instance
        model = Network()
        
        # Add layers to the model
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Compile the model
        model.compile(loss='binary_cross_entropy', optimizer='adam')
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32)

Example:
    Below is a simple example demonstrating how to build and train a neural network:

        from network import Network
        from layers import Dense

        # Create a network instance
        model = Network()

        # Add layers to the model
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_cross_entropy', optimizer='adam')

        # Train the model
        history = model.fit(X_train, y_train, epochs=20, batch_size=16)

        # Make predictions
        predictions = model.predict(X_train)
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from neural_network.base import BaseLayer, OptimizerProtocol
from neural_network.losses import Loss
from neural_network.optimizers import SGD, Momentum, Adam

class Network:
    """
    Neural Network implementation for training and inference.
    
    Provides methods to:
    - Add layers dynamically
    - Compile the network with loss function and optimizer
    - Train on batched data with optional validation
    - Make predictions
    
    Attributes:
        layers (List[BaseLayer]): Ordered list of network layers.
        loss_func (Callable): Loss function for computing training loss.
        loss_grad (Callable): Gradient of the loss function.
        optimizer (OptimizerProtocol): Optimization algorithm for weight updates.
        input_shape (Tuple[int, ...]): Shape of the network's input data.
    """
    def __init__(self):
        self.layers: List[BaseLayer] = []
        self.loss_func = None
        self.loss_grad = None
        self.optimizer: Optional[OptimizerProtocol] = None
        self.input_shape = None

    def add(
            self, 
            layer: BaseLayer
    ) -> None:
        """
        Add a layer to the network.
        
        Args:
            layer (BaseLayer): An instance of a layer that inherits from BaseLayer.
        
        Usage:
            model.add(Dense(units=64, activation='relu'))
        """
        self.layers.append(layer)
        
    def _build(
            self, 
            input_shape: Tuple[int, ...]
    ) -> None:
        """
        Build all layers with the given input shape.
        
        This method initializes the layers with the specified input shape 
        and updates the current shape to match the output shape of 
        the last layer.
        
        Args:
            input_shape (Tuple[int, ...]): Shape of the input data.
        """
        self.input_shape = input_shape
        current_shape = input_shape
        
        for layer in self.layers:
            layer.build(current_shape)
            current_shape = layer.output_shape

    def compile(
            self, 
            loss: str = 'binary_cross_entropy', 
            optimizer: Union[str, OptimizerProtocol] = 'sgd',
            optimizer_params: dict = None
    ) -> None:
        """
        Compile the network with the specified loss function and optimizer.
        
        This method sets the loss function and optimizer for the network, 
        allowing the model to be trained.
        
        Args:
            loss (str): The name of the loss function to use (default is 'binary_cross_entropy').
            optimizer (Union[str, OptimizerProtocol]): The optimizer to use (default is 'sgd').
            optimizer_params (dict): Additional parameters for the optimizer.
        
        Raises:
            ValueError: If the loss function or optimizer is not recognized.
        
        Usage:
            model.compile(loss='mean_squared_error', optimizer='adam')
        """
        # Set loss function
        loss_func = getattr(Loss, loss, None)
        if loss_func is None:
            raise ValueError(f"Unknown loss function: {loss}")
        self.loss_func = loss_func
        self.loss_grad = getattr(Loss, loss + '_gradient')

        # Set optimizer
        if isinstance(optimizer, str):
            optimizer_params = optimizer_params or {}
            if optimizer.lower() == 'sgd':
                self.optimizer = SGD(**optimizer_params)
            elif optimizer.lower() == 'momentum':
                self.optimizer = Momentum(**optimizer_params)
            elif optimizer.lower() == 'adam':
                self.optimizer = Adam(**optimizer_params)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer
        
        # Setup optimizer
        self.optimizer.setup(self.layers)

    def _forward(
            self, 
            X: np.ndarray, 
            training: bool = True
    ) -> np.ndarray:
        """
        Perform a forward pass through all layers.
        
        This method computes the output of the network by passing the input
        data through each layer in sequence.
        
        Args:
            X (np.ndarray): Input data to the network.
            training (bool): Indicates whether the network is in training mode (default is True).
        
        Returns:
            np.ndarray: The output of the network after the forward pass.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output
    
    def _backward(
            self, 
            gradient: np.ndarray
    ) -> None:
        """
        Perform a backward pass through all layers.
        
        This method computes the gradients of the loss with respect to each layer's 
        parameters, enabling weight updates during training.
        
        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the output.
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def _update(
        self
    ) -> None:
        """
        Update weights using the chosen optimizer.
        
        This method applies the optimizer's update rules to adjust the weights 
        of the layers based on the computed gradients.
        """
        for layer in self.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                self.optimizer.update(layer, 'weights', layer.gradients['weights'])
                if layer.use_bias:
                    self.optimizer.update(layer, 'bias', layer.gradients['bias'])
            elif hasattr(layer, 'gamma') and layer.gamma is not None:
                self.optimizer.update(layer, 'gamma', layer.gradients['gamma'])
                self.optimizer.update(layer, 'beta', layer.gradients['beta'])


    def _train_on_batch(
            self, 
            X: np.ndarray, 
            y: np.ndarray
    ) -> float:
        """
        Train the network on a single batch of data.
        
        This method performs the forward pass, computes the loss, performs 
        the backward pass, and updates the weights for the given batch of data.
        
        Args:
            X (np.ndarray): Input data for the batch.
            y (np.ndarray): Target output for the batch.
        
        Returns:
            float: The computed loss for the batch.
        """
        # Forward pass
        pred = self._forward(X, training=True)
        
        # Compute loss
        loss = self.loss_func(y, pred)
        
        # Backward pass
        gradient = self.loss_grad(y, pred)
        self._backward(gradient)
        
        # Update weights
        self._update()
        
        return loss

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions using the trained network.
        
        This method performs a forward pass through the network to 
        generate predictions for the given input data.
        
        Args:
            X (np.ndarray): Input data for which predictions are to be made.
        
        Returns:
            np.ndarray: The predicted output.
        """
        return self._forward(X, training=False)
    
    def _calculate_accuracy(
            self, 
            y_true: np.ndarray, 
            y_pred: np.ndarray
    ) -> float:
        """
        Calculate the accuracy of predictions compared to the true labels.
        
        This method computes the accuracy of the predicted outputs 
        against the true labels for evaluation purposes.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        
        Returns:
            float: The computed accuracy as a fraction of correct predictions.
        """
        if y_true.shape[1] == 1:  # Binary classification
            label = (y_pred > 0.5).astype(int)
        else:  # Multiclass classification
            label = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
        return np.mean(label == y_true)

    def fit(
            self, 
            X: np.ndarray,
            y: np.ndarray, 
            epochs: int = 10,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> dict:
        """
        Train the network on the given data.
        
        This method iterates over the specified number of epochs, shuffles 
        the training data, and processes it in batches, updating the model 
        weights and recording the training history.
        
        Args:
            X (np.ndarray): Input training data.
            y (np.ndarray): Target training labels.
            epochs (int): Number of training epochs (default is 10).
            batch_size (int): Size of each training batch (default is 32).
            validation_data (Optional[Tuple[np.ndarray, np.ndarray]]): 
                Data for validation during training, if any.
        
        Returns:
            history (dict): A dictionary containing the training history (loss and accuracy).
        """
        if self.input_shape is None:
            self._build(input_shape=(X.shape[0], *X.shape[1:]))

        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [] if validation_data is not None else None,
            'val_accuracy': [] if validation_data is not None else None
        }

        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Train on batches
            epoch_loss = 0
            epoch_predictions = []

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                            
                X_batch = X_shuffled[start_idx: end_idx]
                y_batch = y_shuffled[start_idx: end_idx]

                batch_loss = self._train_on_batch(X_batch, y_batch)
                epoch_loss += batch_loss

                # Collect predictions for accuracy calculation
                batch_predictions = self._forward(X_batch, training=False)
                epoch_predictions.append(batch_predictions)

            # Calculate training metrics
            epoch_loss /= n_batches
            epoch_predictions = np.vstack(epoch_predictions)
            train_accuracy = self._calculate_accuracy(y_shuffled, epoch_predictions)
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(train_accuracy)
            
            # Validation
            if validation_data is not None:
                val_X, val_y = validation_data
                val_predictions = self._forward(val_X, training=False)
                val_loss = self.loss_func(val_y, val_predictions)
                val_accuracy = self._calculate_accuracy(val_y, val_predictions)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                    f"loss: {epoch_loss:.4f} - "
                    f"accuracy: {train_accuracy:.4f} - "
                    f"val_loss: {val_loss:.4f} - "
                    f"val_accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                    f"loss: {epoch_loss:.4f} - "
                    f"accuracy: {train_accuracy:.4f}")
                

        return history


