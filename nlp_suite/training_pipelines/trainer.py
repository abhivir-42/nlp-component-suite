"""
Trainer module for model training and evaluation.

This module provides a basic training pipeline that handles the training loop,
optimization, and visualization of training metrics.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nlp_suite.nn_core.value import Value
from nlp_suite.architectures.mlp import Module


class Optimizer:
    """
    Base class for optimizers.
    
    This class defines the interface for optimizers that update model parameters
    based on gradients.
    """

    def __init__(self, parameters: List[Value]) -> None:
        """
        Initialize the optimizer with the parameters to optimize.
        
        Args:
            parameters: List of parameters to optimize
        """
        self.parameters = parameters

    def zero_grad(self) -> None:
        """Zero out the gradients of all parameters."""
        for p in self.parameters:
            p.grad = 0.0

    def step(self) -> None:
        """
        Update parameters based on their gradients.
        
        This method should be overridden by subclasses.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    This implements vanilla SGD with optional momentum and weight decay.
    """

    def __init__(
        self,
        parameters: List[Value],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Initialize the SGD optimizer.
        
        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (default: 0.0, i.e., no momentum)
            weight_decay: Weight decay (L2 penalty) (default: 0.0)
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [0.0] * len(parameters)

    def step(self) -> None:
        """Update parameters using SGD with momentum and weight decay."""
        for i, p in enumerate(self.parameters):
            # Apply weight decay if specified
            if self.weight_decay > 0:
                p.grad += self.weight_decay * p.data
                
            # Apply momentum if specified
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
                p.data += self.velocity[i]
            else:
                p.data -= self.lr * p.grad


class Trainer:
    """
    Trainer class for model training and evaluation.
    
    This class handles the training loop, optimization, and tracking of metrics.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[List[Value], List[Union[Value, float]]], Value],
        metrics: Optional[Dict[str, Callable[[List[Value], List[Union[Value, float]]], float]]] = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use for parameter updates
            loss_fn: The loss function to minimize
            metrics: Additional metrics to track during training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or {}
        
        # Training history
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        for metric_name in self.metrics:
            self.history[metric_name] = []
            self.history[f"val_{metric_name}"] = []

    def train_step(
        self, X: List[List[Union[Value, float]]], y: List[List[Union[Value, float]]]
    ) -> Dict[str, float]:
        """
        Perform a single training step (forward and backward pass).
        
        Args:
            X: Batch of input features
            y: Batch of target values
            
        Returns:
            Dictionary of metrics for this step
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        batch_loss = 0.0
        batch_metrics: Dict[str, float] = {name: 0.0 for name in self.metrics}
        
        # Process each example in batch
        for x_i, y_i in zip(X, y):
            # Forward pass
            y_pred = self.model(x_i)
            
            # Compute loss
            loss = self.loss_fn(y_pred, y_i)
            batch_loss += loss.data
            
            # Compute metrics
            for metric_name, metric_fn in self.metrics.items():
                batch_metrics[metric_name] += metric_fn(y_pred, y_i)
            
            # Backward pass
            loss.backward()
        
        # Average metrics over batch
        batch_size = len(X)
        batch_loss /= batch_size
        for metric_name in batch_metrics:
            batch_metrics[metric_name] /= batch_size
        
        # Update parameters
        self.optimizer.step()
        
        return {"loss": batch_loss, **batch_metrics}

    def evaluate(
        self, X: List[List[Union[Value, float]]], y: List[List[Union[Value, float]]]
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            X: Validation input features
            y: Validation target values
            
        Returns:
            Dictionary of validation metrics
        """
        val_loss = 0.0
        val_metrics: Dict[str, float] = {name: 0.0 for name in self.metrics}
        
        # Process each example
        for x_i, y_i in zip(X, y):
            # Forward pass
            y_pred = self.model(x_i)
            
            # Compute loss
            loss = self.loss_fn(y_pred, y_i)
            val_loss += loss.data
            
            # Compute metrics
            for metric_name, metric_fn in self.metrics.items():
                val_metrics[metric_name] += metric_fn(y_pred, y_i)
        
        # Average metrics
        dataset_size = len(X)
        val_loss /= dataset_size
        for metric_name in val_metrics:
            val_metrics[metric_name] /= dataset_size
        
        return {"val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}

    def train(
        self,
        X_train: List[List[Union[Value, float]]],
        y_train: List[List[Union[Value, float]]],
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: Optional[Tuple[List[List[Union[Value, float]]], List[List[Union[Value, float]]]]] = None,
        verbose: int = 1,
        callbacks: List[Any] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for a fixed number of epochs.
        
        Args:
            X_train: Training input features
            y_train: Training target values
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_data: Optional validation data for evaluation after each epoch
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            callbacks: List of callbacks to apply during training
            
        Returns:
            Training history dictionary
        """
        # Reset history
        self.history = {"loss": [], "val_loss": []}
        for metric_name in self.metrics:
            self.history[metric_name] = []
            self.history[f"val_{metric_name}"] = []
        
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Create batches
            indices = np.random.permutation(n_samples)
            X_shuffled = [X_train[i] for i in indices]
            y_shuffled = [y_train[i] for i in indices]
            
            # Training loop
            epoch_loss = 0.0
            epoch_metrics: Dict[str, float] = {name: 0.0 for name in self.metrics}
            
            # Create progress bar if verbose
            iterator = range(n_batches)
            if verbose == 1:
                iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{epochs}")
                
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Train on batch
                batch_metrics = self.train_step(batch_X, batch_y)
                
                # Update epoch metrics
                epoch_loss += batch_metrics["loss"] * (end_idx - start_idx) / n_samples
                for metric_name in self.metrics:
                    epoch_metrics[metric_name] += batch_metrics[metric_name] * (end_idx - start_idx) / n_samples
            
            # Update history
            self.history["loss"].append(epoch_loss)
            for metric_name, value in epoch_metrics.items():
                self.history[metric_name].append(value)
            
            # Evaluate on validation data if provided
            val_metrics = {}
            if validation_data is not None:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val)
                for metric_name, value in val_metrics.items():
                    self.history[metric_name].append(value)
            
            # Print epoch summary
            if verbose > 0:
                elapsed_time = time.time() - start_time
                metrics_str = f"loss: {epoch_loss:.4f}"
                for metric_name, value in epoch_metrics.items():
                    metrics_str += f" - {metric_name}: {value:.4f}"
                if validation_data is not None:
                    metrics_str += f" - val_loss: {val_metrics['val_loss']:.4f}"
                    for metric_name in self.metrics:
                        val_metric_name = f"val_{metric_name}"
                        metrics_str += f" - {val_metric_name}: {val_metrics[val_metric_name]:.4f}"
                print(f"Epoch {epoch+1}/{epochs} - {elapsed_time:.2f}s - {metrics_str}")
        
        return self.history

    def plot_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot the training history.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history["loss"], label="Training Loss")
        if "val_loss" in self.history:
            plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot metrics if available
        if self.metrics:
            plt.subplot(1, 2, 2)
            for metric_name in self.metrics:
                plt.plot(self.history[metric_name], label=f"Training {metric_name}")
                if f"val_{metric_name}" in self.history:
                    plt.plot(self.history[f"val_{metric_name}"], label=f"Validation {metric_name}")
            plt.title("Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show() 