"""
Multi-Layer Perceptron implementation using the autograd engine.

This module provides a clean, modular implementation of a Multi-Layer Perceptron
using the autograd Value class.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple, Union

import numpy as np

from nlp_suite.nn_core.value import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        """
        Zero out the gradients of all parameters in the module.
        
        Should be called before backward() to reset gradients from previous passes.
        """
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        """
        Return a list of all trainable parameters in the module.
        
        This method should be overridden by subclasses.
        
        Returns:
            List of trainable parameters (Value objects)
        """
        return []


class Neuron(Module):
    """
    A single neuron with multiple inputs and one output.
    
    This implements a basic neuron that computes y = activation(w·x + b)
    where w are weights, x is the input, b is the bias, and activation
    is an optional non-linear function.
    """

    def __init__(
        self, nin: int, activation: Callable[[Value], Value] = lambda x: x
    ) -> None:
        """
        Initialize a neuron with the specified number of inputs and activation function.
        
        Args:
            nin: Number of input features
            activation: Activation function to apply (default: identity function)
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation

    def __call__(self, x: List[Union[Value, float]]) -> Value:
        """
        Perform the forward pass computation for the neuron.
        
        Args:
            x: Input values
            
        Returns:
            Output of the neuron after applying weights, bias and activation
        """
        # Convert inputs to Value objects if they aren't already
        x_values = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        
        # Check input dimension
        assert len(x_values) == len(self.w), f"Expected {len(self.w)} inputs, got {len(x_values)}"
        
        # Compute weighted sum
        act = sum((wi * xi for wi, xi in zip(self.w, x_values)), self.b)
        
        # Apply activation function
        return self.activation(act)

    def parameters(self) -> List[Value]:
        """
        Return a list of the neuron's trainable parameters.
        
        Returns:
            List containing weights and bias
        """
        return self.w + [self.b]


class Layer(Module):
    """
    A layer of neurons, each taking the same input.
    
    This implements a fully-connected layer of neurons.
    """

    def __init__(
        self, nin: int, nout: int, activation: Callable[[Value], Value] = lambda x: x
    ) -> None:
        """
        Initialize a layer with the specified dimensions and activation function.
        
        Args:
            nin: Number of input features
            nout: Number of output features (neurons)
            activation: Activation function to apply to each neuron's output
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: List[Union[Value, float]]) -> List[Value]:
        """
        Perform the forward pass computation for the layer.
        
        Args:
            x: Input values
            
        Returns:
            List of outputs from each neuron in the layer
        """
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        """
        Return a list of all trainable parameters in the layer.
        
        Returns:
            List of all parameters from all neurons in the layer
        """
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """
    Multi-Layer Perceptron implementation.
    
    This class implements a multi-layer perceptron (feedforward neural network)
    with configurable layer sizes and activation functions.
    """

    def __init__(
        self,
        nin: int,
        nouts: List[int],
        activation: Callable[[Value], Value] = lambda x: x.relu(),
        activation_final: Callable[[Value], Value] = lambda x: x,
    ) -> None:
        """
        Initialize an MLP with the specified architecture.
        
        Args:
            nin: Number of input features
            nouts: List of neurons per layer (including output layer)
            activation: Activation function to use for hidden layers
            activation_final: Activation function for final layer (default: identity)
        """
        sz = [nin] + nouts
        self.layers = []
        
        # Create all layers except the last one with the specified activation
        for i in range(len(nouts) - 1):
            self.layers.append(Layer(sz[i], sz[i + 1], activation))
        
        # Create the last layer with the final activation
        if nouts:
            self.layers.append(Layer(sz[-2], sz[-1], activation_final))

    def __call__(self, x: List[Union[Value, float]]) -> List[Value]:
        """
        Perform the forward pass through the entire MLP.
        
        Args:
            x: Input values
            
        Returns:
            Output values from the final layer
        """
        # Handle scalar inputs by converting to list
        if not isinstance(x, list):
            x = [x]
            
        # Forward pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        """
        Return a list of all trainable parameters in the MLP.
        
        Returns:
            List containing all parameters from all layers
        """
        return [p for layer in self.layers for p in layer.parameters()] 