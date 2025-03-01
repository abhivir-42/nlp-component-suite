"""
Integration tests for the model serving API.

These tests validate that the FastAPI server correctly serves models
and handles requests and responses properly.
"""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from nlp_suite.architectures.mlp import MLP
from nlp_suite.model_serving.api import ModelMetadata, app, model_registry


class TestModelServingAPI(unittest.TestCase):
    """Test the model serving API."""

    def setUp(self):
        """Set up test case with a test client and a mock model."""
        self.client = TestClient(app)
        
        # Reset the model registry before each test
        model_registry.models = {}
        model_registry.metadata = {}
        
        # Create a test model
        self.model = MLP(nin=3, nouts=[4, 2])
        
        # Create test metadata
        self.metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model for API",
            input_shape=[3],
            output_shape=[2],
            created_at="2023-01-01T00:00:00",
        )
        
        # Register the model
        model_registry.register_model("test_model", self.model, self.metadata)

    def test_root_endpoint(self):
        """Test the root endpoint returns the correct response."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("NLP Component Suite API", data["message"])

    def test_list_models_endpoint(self):
        """Test that the models endpoint lists all available models."""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "test_model")
        self.assertEqual(data[0]["version"], "1.0.0")
        self.assertEqual(data[0]["input_shape"], [3])
        self.assertEqual(data[0]["output_shape"], [2])

    def test_get_model_metadata_endpoint(self):
        """Test getting metadata for a specific model."""
        response = self.client.get("/models/test_model")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "test_model")
        self.assertEqual(data["description"], "Test model for API")

    def test_predict_endpoint(self):
        """Test that the predict endpoint returns correct predictions."""
        # Create a prediction request
        request_data = {
            "inputs": [[0.1, 0.2, 0.3]],
            "model_name": "test_model"
        }
        
        response = self.client.post(
            "/predict",
            json=request_data,
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertIn("predictions", data)
        self.assertIn("model_name", data)
        self.assertIn("prediction_time_ms", data)
        
        # Verify the model name
        self.assertEqual(data["model_name"], "test_model")
        
        # Check predictions format (should be a list of lists of floats)
        self.assertEqual(len(data["predictions"]), 1)  # One input, one prediction
        self.assertEqual(len(data["predictions"][0]), 2)  # Output size is 2
        
        # Prediction time should be a positive number
        self.assertGreater(data["prediction_time_ms"], 0)

    def test_predict_endpoint_with_nonexistent_model(self):
        """Test that requesting a non-existent model returns an error."""
        request_data = {
            "inputs": [[0.1, 0.2, 0.3]],
            "model_name": "nonexistent_model"
        }
        
        response = self.client.post(
            "/predict",
            json=request_data,
        )
        
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    def test_model_processing_headers(self):
        """Test that the API includes processing time headers."""
        response = self.client.get("/")
        self.assertIn("X-Process-Time", response.headers)
        process_time = float(response.headers["X-Process-Time"])
        self.assertGreaterEqual(process_time, 0)


if __name__ == "__main__":
    unittest.main() 