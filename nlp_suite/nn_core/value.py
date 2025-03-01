"""
Value class for autograd engine.

This module provides a foundational Value class that tracks gradients
and implements automatic differentiation, inspired by micrograd but
with additional compatibility with PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import numpy as np
from typing_extensions import Self


class Value:
    """
    A scalar value with support for automatic differentiation.

    This class implements a scalar value that can track its computation
    history and compute gradients through backpropagation. It's designed
    to be compatible with PyTorch's autograd system.

    Attributes:
        data: The scalar value.
        grad: The gradient of the loss with respect to this value.
        _backward: Function that computes the gradient contribution.
        _prev: Set of parent Value objects that were used to compute this value.
        _op: String representation of the operation that created this value.
    """

    def __init__(
        self,
        data: Union[float, int],
        _children: Tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        """
        Initialize a Value object.

        Args:
            data: The scalar value to be wrapped.
            _children: Parent Value objects that were used to compute this value.
            _op: String representation of the operation that created this value.
            label: Optional label for visualization purposes.
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        """Return string representation of the Value."""
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union[Value, float, int]) -> Value:
        """Add two Values or a Value and a scalar."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union[Value, float, int]) -> Value:
        """Multiply two Values or a Value and a scalar."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: Union[float, int]) -> Value:
        """Raise Value to a power (where power is a scalar)."""
        assert isinstance(other, (int, float)), "Only supporting int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> Value:
        """Add a scalar and a Value."""
        return self + other

    def __rmul__(self, other: Union[float, int]) -> Value:
        """Multiply a scalar and a Value."""
        return self * other

    def __neg__(self) -> Value:
        """Negate a Value."""
        return self * -1

    def __sub__(self, other: Union[Value, float, int]) -> Value:
        """Subtract two Values or a Value and a scalar."""
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> Value:
        """Subtract a Value from a scalar."""
        return other + (-self)

    def __truediv__(self, other: Union[Value, float, int]) -> Value:
        """Divide a Value by another Value or scalar."""
        return self * (other**-1)

    def __rtruediv__(self, other: Union[float, int]) -> Value:
        """Divide a scalar by a Value."""
        return other * (self**-1)

    def tanh(self) -> Value:
        """Apply tanh activation function."""
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Compute the exponential of a Value."""
        x = self.data
        exp_x = math.exp(x)
        out = Value(exp_x, (self,), "exp")

        def _backward() -> None:
            self.grad += exp_x * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Compute the natural logarithm of a Value."""
        x = self.data
        assert x > 0, "logarithm is only defined for positive values"
        out = Value(math.log(x), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / x) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Apply ReLU activation function."""
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """Apply sigmoid activation function."""
        sig = 1 / (1 + math.exp(-self.data))
        out = Value(sig, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Perform backpropagation to compute gradients for all values in the computation graph.

        This method uses topological sort to ensure that nodes are processed
        in the correct order during backpropagation.
        """
        # Topological order of all nodes in the computation graph
        topo = []
        visited = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set the gradient of the output to 1
        self.grad = 1.0

        # Backpropagate the gradients in reverse topological order
        for node in reversed(topo):
            node._backward() 