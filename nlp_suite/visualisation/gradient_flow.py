"""
Gradient Flow Visualization module.

This module provides tools to visualize gradient flow through neural networks,
which is essential for diagnosing issues like vanishing/exploding gradients.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from nlp_suite.nn_core.value import Value
from nlp_suite.architectures.mlp import Module


def plot_grad_flow_bars(
    named_parameters: Dict[str, List[Value]], figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the gradient flow in the form of a bar plot for each layer's gradients.
    
    This visualization helps to identify layers where gradients are vanishing
    or exploding.
    
    Args:
        named_parameters: Dictionary mapping layer names to lists of parameters
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, params in named_parameters.items():
        # Skip layers without parameters
        if not params:
            continue
            
        # Add layer name
        layers.append(name)
        
        # Calculate average and max of absolute gradient values
        ave_grad = np.mean([abs(p.grad) for p in params])
        max_grad = np.max([abs(p.grad) for p in params])
        ave_grads.append(ave_grad)
        max_grads.append(max_grad)
    
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.bar(range(len(max_grads)), max_grads, alpha=0.5, lw=1, color="r")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0)
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([
        "Average gradient magnitude",
        "Max gradient magnitude",
    ], loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_grad_flow_lines(
    named_parameters: Dict[str, List[Value]], figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the gradient flow in the form of a line plot for each layer's gradients.
    
    This visualization helps to identify patterns in gradient flow across layers.
    
    Args:
        named_parameters: Dictionary mapping layer names to lists of parameters
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    ave_grads = []
    layers = []
    
    for name, params in named_parameters.items():
        # Skip layers without parameters
        if not params:
            continue
            
        # Add layer name
        layers.append(name)
        
        # Calculate average of absolute gradient values
        ave_grad = np.mean([abs(p.grad) for p in params])
        ave_grads.append(ave_grad)
    
    plt.plot(range(len(ave_grads)), ave_grads, "o-", lw=2)
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_param_magnitudes(
    named_parameters: Dict[str, List[Value]], figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the parameter magnitudes across layers.
    
    This visualization helps to identify if any layers have unusually large
    or small parameter values.
    
    Args:
        named_parameters: Dictionary mapping layer names to lists of parameters
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    ave_params = []
    max_params = []
    layers = []
    
    for name, params in named_parameters.items():
        # Skip layers without parameters
        if not params:
            continue
            
        # Add layer name
        layers.append(name)
        
        # Calculate average and max of absolute parameter values
        ave_param = np.mean([abs(p.data) for p in params])
        max_param = np.max([abs(p.data) for p in params])
        ave_params.append(ave_param)
        max_params.append(max_param)
    
    plt.bar(range(len(ave_params)), ave_params, alpha=0.5, lw=1, color="g")
    plt.bar(range(len(max_params)), max_params, alpha=0.5, lw=1, color="y")
    plt.hlines(0, 0, len(ave_params) + 1, lw=2, color="k")
    plt.xticks(range(len(ave_params)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_params))
    plt.ylim(bottom=0)
    plt.xlabel("Layers")
    plt.ylabel("Parameter Magnitude")
    plt.title("Parameter Magnitudes")
    plt.grid(True)
    plt.legend([
        "Average parameter magnitude",
        "Max parameter magnitude",
    ], loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_gradient_ratios(
    named_parameters: Dict[str, List[Value]], figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the ratio of gradient to parameter magnitudes.
    
    This visualization helps to identify if any layers have gradients that
    are disproportionately large or small compared to their parameter values.
    
    Args:
        named_parameters: Dictionary mapping layer names to lists of parameters
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    ratios = []
    layers = []
    
    for name, params in named_parameters.items():
        # Skip layers without parameters
        if not params:
            continue
            
        # Add layer name
        layers.append(name)
        
        # Calculate ratio of average gradient to average parameter magnitude
        ave_grad = np.mean([abs(p.grad) for p in params])
        ave_param = np.mean([abs(p.data) for p in params])
        # Avoid division by zero
        ratio = ave_grad / max(ave_param, 1e-10)
        ratios.append(ratio)
    
    plt.bar(range(len(ratios)), ratios, alpha=0.7, lw=1, color="purple")
    plt.hlines(0, 0, len(ratios) + 1, lw=2, color="k")
    plt.xticks(range(len(ratios)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ratios))
    plt.ylim(bottom=0)
    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("Gradient to Parameter Ratio (log scale)")
    plt.title("Gradient to Parameter Magnitude Ratios")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_computational_graph(
    root: Value,
    max_depth: int = 10,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Visualize the computational graph starting from a root Value.
    
    Args:
        root: Root Value node to start visualization from
        max_depth: Maximum depth to traverse in the graph
        figsize: Figure size
    """
    try:
        import networkx as nx
    except ImportError:
        print("Please install networkx with 'pip install networkx' to use this feature")
        return
    
    plt.figure(figsize=figsize)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Keep track of nodes to avoid cycles
    visited = set()
    
    def add_nodes_edges(node: Value, depth: int = 0) -> None:
        """Recursively add nodes and edges to the graph."""
        if depth > max_depth or node in visited:
            return
            
        visited.add(node)
        
        # Add node with attributes
        node_id = id(node)
        G.add_node(
            node_id,
            label=f"{node._op if node._op else 'input'}\n{node.data:.4f}\n(grad: {node.grad:.4f})",
            data=node.data,
            grad=node.grad,
            op=node._op,
        )
        
        # Add edges to children
        for child in node._prev:
            child_id = id(child)
            add_nodes_edges(child, depth + 1)
            G.add_edge(child_id, node_id)
    
    # Build the graph
    add_nodes_edges(root)
    
    # Layout the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with colors based on gradients
    node_colors = [abs(G.nodes[n]["grad"]) for n in G.nodes]
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}
    
    # Normalize colors
    if node_colors:
        vmax = max(node_colors)
        node_colors = [c / max(vmax, 1e-10) for c in node_colors]
    
    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        node_size=1500,
        font_size=8,
        font_color="black",
        font_weight="bold",
        arrows=True,
        connectionstyle="arc3,rad=0.1",
    )
    
    plt.title("Computational Graph Visualization")
    plt.tight_layout()
    plt.show()


def extract_named_parameters(model: Module) -> Dict[str, List[Value]]:
    """
    Extract named parameters from a model for visualization.
    
    This is a helper function to organize parameters by layer for visualization.
    
    Args:
        model: The model to extract parameters from
        
    Returns:
        Dictionary mapping layer names to lists of parameters
    """
    named_parameters = {}
    
    # Special handling for MLP from our architecture module
    if hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            named_parameters[f"layer_{i}"] = layer.parameters()
    else:
        # Generic fallback
        named_parameters["model"] = model.parameters()
    
    return named_parameters


def visualize_model_gradients(
    model: Module, loss_root: Optional[Value] = None
) -> None:
    """
    Visualize gradients across all layers of a model.
    
    This is a convenience function that extracts named parameters and
    calls the various visualization functions.
    
    Args:
        model: The model to visualize gradients for
        loss_root: Optional loss Value to visualize computation graph
    """
    named_parameters = extract_named_parameters(model)
    
    print("Gradient Flow Bar Plot")
    plot_grad_flow_bars(named_parameters)
    
    print("Gradient Flow Line Plot")
    plot_grad_flow_lines(named_parameters)
    
    print("Parameter Magnitudes")
    plot_param_magnitudes(named_parameters)
    
    print("Gradient to Parameter Ratios")
    plot_gradient_ratios(named_parameters)
    
    if loss_root is not None:
        print("Computational Graph")
        visualize_computational_graph(loss_root) 