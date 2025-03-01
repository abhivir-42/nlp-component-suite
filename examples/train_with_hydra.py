"""
Hydra-based training script.

This script demonstrates how to use Hydra for configuration management
in machine learning projects.
"""

import os
import logging
from typing import Dict, List, Any

import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from nlp_suite.nn_core.value import Value
from nlp_suite.architectures.mlp import MLP
from nlp_suite.training_pipelines.trainer import SGD, Trainer
from nlp_suite.visualisation.gradient_flow import visualize_model_gradients

# Setup logger
log = logging.getLogger(__name__)


def generate_data(cfg: DictConfig):
    """
    Generate a simple binary classification dataset based on config.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        X: Input features
        y: Target labels
    """
    n_samples = cfg.data.n_samples
    noise = cfg.data.noise
    
    # Generate two clusters
    X = []
    y = []
    
    # First cluster (class 0)
    n_class_0 = n_samples // 2
    x1 = np.random.randn(n_class_0) * noise + 1.5
    x2 = np.random.randn(n_class_0) * noise + 1.5
    for i in range(n_class_0):
        X.append([x1[i], x2[i]])
        y.append([1.0, 0.0])  # One-hot encoding
    
    # Second cluster (class 1)
    n_class_1 = n_samples - n_class_0
    x1 = np.random.randn(n_class_1) * noise - 1.5
    x2 = np.random.randn(n_class_1) * noise - 1.5
    for i in range(n_class_1):
        X.append([x1[i], x2[i]])
        y.append([0.0, 1.0])  # One-hot encoding
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    
    return X, y


def binary_cross_entropy(y_pred, y_true):
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        
    Returns:
        Loss value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-7
    
    # Compute loss for each output
    losses = []
    for i in range(len(y_pred)):
        p = y_pred[i]
        t = y_true[i]
        
        # Clip prediction to avoid numerical issues
        p_clipped = p if p.data > epsilon else Value(epsilon)
        p_clipped = p_clipped if p_clipped.data < 1 - epsilon else Value(1 - epsilon)
        
        # Compute loss term
        loss_term = -t * p_clipped.log() - (1 - t) * (1 - p_clipped).log()
        losses.append(loss_term)
    
    # Sum all loss terms
    total_loss = sum(losses, Value(0))
    
    return total_loss


def accuracy(y_pred, y_true):
    """
    Compute accuracy.
    
    Args:
        y_pred: Predicted probabilities
        y_true: True labels
        
    Returns:
        Accuracy value
    """
    # Get predicted class
    pred_class = np.argmax([p.data for p in y_pred])
    
    # Get true class
    true_class = np.argmax(y_true)
    
    # Return 1 if correct, 0 if incorrect
    return 1.0 if pred_class == true_class else 0.0


def get_activation_fn(name: str):
    """
    Get activation function by name.
    
    Args:
        name: Name of the activation function
        
    Returns:
        Activation function
    """
    if name == "relu":
        return lambda x: x.relu()
    elif name == "sigmoid":
        return lambda x: x.sigmoid()
    elif name == "tanh":
        return lambda x: x.tanh()
    else:
        return lambda x: x  # Identity function


def create_model(cfg: DictConfig):
    """
    Create model based on configuration.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Model instance
    """
    if cfg.model.type == "mlp":
        # Get activation functions
        activation = get_activation_fn(cfg.model.params.activation)
        activation_final = get_activation_fn(cfg.model.params.activation_final)
        
        # Create MLP model
        model = MLP(
            nin=cfg.model.params.nin,
            nouts=cfg.model.params.nouts,
            activation=activation,
            activation_final=activation_final,
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")


def create_optimizer(cfg: DictConfig, parameters):
    """
    Create optimizer based on configuration.
    
    Args:
        cfg: Hydra configuration
        parameters: Model parameters to optimize
        
    Returns:
        Optimizer instance
    """
    if cfg.training.optimizer.type == "sgd":
        return SGD(
            parameters=parameters,
            lr=cfg.training.optimizer.params.lr,
            momentum=cfg.training.optimizer.params.momentum,
            weight_decay=cfg.training.optimizer.params.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.training.optimizer.type}")


def get_loss_fn(name: str):
    """
    Get loss function by name.
    
    Args:
        name: Name of the loss function
        
    Returns:
        Loss function
    """
    if name == "binary_cross_entropy":
        return binary_cross_entropy
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_metrics(metric_names: List[str]) -> Dict[str, Any]:
    """
    Get metrics by name.
    
    Args:
        metric_names: List of metric names
        
    Returns:
        Dictionary mapping metric names to metric functions
    """
    metrics = {}
    for name in metric_names:
        if name == "accuracy":
            metrics[name] = accuracy
        else:
            log.warning(f"Unknown metric: {name}, skipping")
    return metrics


@hydra.main(config_path="../configs", config_name="model/mlp")
def main(cfg: DictConfig):
    """
    Main function to run the training process.
    
    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate data
    X, y = generate_data(cfg)
    
    # Split into train and validation sets
    train_size = int(cfg.data.train_ratio * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    log.info(f"Training set size: {len(X_train)}")
    log.info(f"Validation set size: {len(X_val)}")
    
    # Create model
    model = create_model(cfg)
    log.info(f"Created model: {model}")
    
    # Create optimizer
    optimizer = create_optimizer(cfg, model.parameters())
    log.info(f"Created optimizer: {optimizer}")
    
    # Get loss function
    loss_fn = get_loss_fn(cfg.training.loss_fn)
    
    # Get metrics
    metrics = get_metrics(cfg.training.metrics)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
    # Train the model
    log.info("Starting training...")
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
    )
    log.info("Training completed")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for metric in metrics:
        plt.plot(history[metric], label=f"Training {metric}")
        plt.plot(history[f"val_{metric}"], label=f"Validation {metric}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("plots/training_history.png")
    log.info("Saved training history plot to plots/training_history.png")
    
    # Visualize model gradients
    # First, we need to do a forward and backward pass to compute gradients
    x = X[0]
    y_true = y[0]
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Save model parameters and gradients for inspection
    log.info("Model parameters and gradients:")
    for i, layer in enumerate(model.layers):
        params = layer.parameters()
        log.info(f"Layer {i}: {len(params)} parameters")
        
        # Log a few parameters and their gradients
        for j, p in enumerate(params[:5]):  # Log only first 5 parameters
            log.info(f"  Param {j}: value={p.data:.4f}, grad={p.grad:.4f}")
    
    log.info("Training completed successfully")


if __name__ == "__main__":
    main() 