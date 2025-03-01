"""
Unit tests for the Value class.

This module contains tests for the Value class in the autograd engine.
"""

import math
import unittest

from nlp_suite.nn_core.value import Value


class TestValue(unittest.TestCase):
    """Test cases for the Value class."""

    def test_creation(self):
        """Test creation of a Value object."""
        x = Value(2.0)
        self.assertEqual(x.data, 2.0)
        self.assertEqual(x.grad, 0.0)

    def test_addition(self):
        """Test addition of Value objects."""
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        self.assertEqual(z.data, 5.0)
        
        # Test backward pass
        z.backward()
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, 1.0)

    def test_multiplication(self):
        """Test multiplication of Value objects."""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        self.assertEqual(z.data, 6.0)
        
        # Test backward pass
        z.backward()
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, 2.0)

    def test_power(self):
        """Test power operation on Value objects."""
        x = Value(2.0)
        z = x ** 3
        self.assertEqual(z.data, 8.0)
        
        # Test backward pass
        z.backward()
        self.assertEqual(x.grad, 12.0)  # 3 * (2^2)

    def test_division(self):
        """Test division of Value objects."""
        x = Value(8.0)
        y = Value(2.0)
        z = x / y
        self.assertEqual(z.data, 4.0)
        
        # Test backward pass
        z.backward()
        self.assertEqual(x.grad, 0.5)  # 1/2
        self.assertEqual(y.grad, -2.0)  # -8/(2^2)

    def test_relu(self):
        """Test ReLU activation function."""
        # Test positive input
        x = Value(2.0)
        z = x.relu()
        self.assertEqual(z.data, 2.0)
        
        z.backward()
        self.assertEqual(x.grad, 1.0)
        
        # Test negative input
        x = Value(-2.0)
        z = x.relu()
        self.assertEqual(z.data, 0.0)
        
        x.grad = 0.0  # Reset gradient
        z.backward()
        self.assertEqual(x.grad, 0.0)

    def test_sigmoid(self):
        """Test sigmoid activation function."""
        x = Value(0.0)
        z = x.sigmoid()
        self.assertEqual(z.data, 0.5)
        
        z.backward()
        self.assertEqual(x.grad, 0.25)  # 0.5 * (1 - 0.5)

    def test_tanh(self):
        """Test tanh activation function."""
        x = Value(0.0)
        z = x.tanh()
        self.assertEqual(z.data, 0.0)
        
        z.backward()
        self.assertEqual(x.grad, 1.0)  # 1 - (tanh(0)^2) = 1 - 0 = 1

    def test_complex_expression(self):
        """Test a more complex expression with multiple operations."""
        # Compute: f(x,y) = (x + 2*y) * (x + y)
        x = Value(2.0)
        y = Value(3.0)
        
        # (x + 2*y)
        a = x + 2 * y
        self.assertEqual(a.data, 8.0)
        
        # (x + y)
        b = x + y
        self.assertEqual(b.data, 5.0)
        
        # (x + 2*y) * (x + y)
        c = a * b
        self.assertEqual(c.data, 40.0)
        
        # Compute gradients
        c.backward()
        
        # Verify gradients
        # ∂f/∂x = (1 * (x + y)) + ((x + 2*y) * 1) = (x + y) + (x + 2*y) = 2*x + 3*y
        # For x=2, y=3: ∂f/∂x = 2*2 + 3*3 = 4 + 9 = 13
        self.assertEqual(x.grad, 13.0)
        
        # ∂f/∂y = (2 * (x + y)) + ((x + 2*y) * 1) = 2*(x + y) + (x + 2*y) = 2*x + 2*y + x + 2*y = 3*x + 4*y
        # For x=2, y=3: ∂f/∂y = 3*2 + 4*3 = 6 + 12 = 18
        self.assertEqual(y.grad, 18.0)


if __name__ == "__main__":
    unittest.main() 