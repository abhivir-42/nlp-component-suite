"""
Simple MLP Example.

This script demonstrates how to use the MLP implementation to solve
a simple binary classification problem.
"""

import numpy as np
import matplotlib.pyplot as plt

from nlp_suite.nn_core.value import Value
from nlp_suite.architectures.mlp import MLP
from nlp_suite.training_pipelines.trainer import SGD, Trainer
from nlp_suite.visualisation.gradient_flow import visualize_model_gradients


def generate_data(n_samples=100, noise=0.1):
    """
    Generate a simple binary classification dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        
    Returns:
        X: Input features
        y: Target labels
    """
    # Generate two clusters
    X = []
    y = []
    
    # First cluster (class 0)
    n_class_0 = n_samples // 2
    x1 = np.random.randn(n_class_0) * 0.5 + 1.5
    x2 = np.random.randn(n_class_0) * 0.5 + 1.5
    for i in range(n_class_0):
        X.append([x1[i], x2[i]])
        y.append([1.0, 0.0])  # One-hot encoding
    
    # Second cluster (class 1)
    n_class_1 = n_samples - n_class_0
    x1 = np.random.randn(n_class_1) * 0.5 - 1.5
    x2 = np.random.randn(n_class_1) * 0.5 - 1.5
    for i in range(n_class_1):
        X.append([x1[i], x2[i]])
        y.append([0.0, 1.0])  # One-hot encoding
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    
    return X, y


def plot_data(X, y, model=None):
    """
    Plot the data and decision boundary if model is provided.
    
    Args:
        X: Input features
        y: Target labels
        model: Optional model to plot decision boundary
    """
    plt.figure(figsize=(10, 8))
    
    # Extract class labels
    y_class = [np.argmax(yi) for yi in y]
    
    # Plot data points
    plt.scatter([x[0] for i, x in enumerate(X) if y_class[i] == 0],
                [x[1] for i, x in enumerate(X) if y_class[i] == 0],
                c='blue', label='Class 0')
    plt.scatter([x[0] for i, x in enumerate(X) if y_class[i] == 1],
                [x[1] for i, x in enumerate(X) if y_class[i] == 1],
                c='red', label='Class 1')
    
    # Plot decision boundary if model is provided
    if model:
        # Create a grid of points
        x_min, x_max = min([x[0] for x in X]) - 1, max([x[0] for x in X]) + 1
        y_min, y_max = min([x[1] for x in X]) - 1, max([x[1] for x in X]) + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Predict class for each point in the grid
        Z = []
        for i in range(xx.shape[0]):
            row = []
            for j in range(xx.shape[1]):
                point = [xx[i, j], yy[i, j]]
                pred = model(point)
                row.append(np.argmax([p.data for p in pred]))
            Z.append(row)
        Z = np.array(Z)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Classification Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()


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


def main():
    """Run the example."""
    # Generate data
    X, y = generate_data(n_samples=200, noise=0.2)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Plot the data
    plot_data(X, y)
    
    # Create model
    model = MLP(
        nin=2,
        nouts=[4, 2],
        activation=lambda x: x.tanh(),
        activation_final=lambda x: x.sigmoid(),
    )
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=binary_cross_entropy,
        metrics={"accuracy": accuracy},
    )
    
    # Train the model
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
    )
    
    # Plot training history
    trainer.plot_history()
    
    # Plot data with decision boundary
    plot_data(X, y, model)
    
    # Visualize gradients
    # First, we need to do a forward and backward pass to compute gradients
    x = X[0]
    y_true = y[0]
    y_pred = model(x)
    loss = binary_cross_entropy(y_pred, y_true)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Visualize gradients
    visualize_model_gradients(model, loss)


if __name__ == "__main__":
    main() 