"""
End-to-end tests for the complete model workflow.

These tests validate the entire pipeline from training to serving,
ensuring all components work together correctly in a real-world scenario.
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
from fastapi.testclient import TestClient

from nlp_suite.architectures.mlp import MLP
from nlp_suite.model_serving.api import app, load_model_from_path
from nlp_suite.training_pipelines.trainer import Trainer


class TestModelTrainingAndServing(unittest.TestCase):
    """End-to-end test for the complete model workflow."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create temporary directories
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.temp_dir, "models")
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Generate synthetic dataset
        np.random.seed(42)
        cls.X = np.random.randn(200, 3).tolist()
        cls.y = [[1, 0] if sum(x) > 0 else [0, 1] for x in cls.X]
        
        # Split into train and test sets
        cls.X_train, cls.X_test = cls.X[:160], cls.X[160:]
        cls.y_train, cls.y_test = cls.y[:160], cls.y[160:]
        
        # Train the model
        cls.model = MLP(
            nin=3,
            nouts=[4, 2],
            activation=lambda x: x.tanh(),
            activation_final=lambda x: x.sigmoid(),
        )
        cls.trainer = Trainer(
            model=cls.model,
            learning_rate=0.1,
            epochs=10,
            batch_size=32,
        )
        cls.trainer.train(cls.X_train, cls.y_train)
        
        # Save the model
        cls.model_path = os.path.join(cls.model_dir, "test_model.pth")
        cls.trainer.save_model(cls.model_path)
        
        # Create API test client
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests are done."""
        shutil.rmtree(cls.temp_dir)

    def test_model_training_accuracy(self):
        """Test that the trained model achieves reasonable accuracy."""
        # Evaluate the model
        metrics = self.trainer.evaluate(self.X_test, self.y_test)
        
        # Check accuracy
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.7)  # Expect at least 70% accuracy

    def test_model_loading_and_serving(self):
        """Test that the saved model can be loaded and served via the API."""
        # Load the model into the API
        model_name = load_model_from_path(self.model_path, "test_e2e_model")
        
        # Check model is registered
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        model_names = [model["name"] for model in models]
        self.assertIn(model_name, model_names)
        
        # Make a prediction request
        test_sample = self.X_test[0]
        request_data = {
            "inputs": [test_sample],
            "model_name": model_name,
        }
        
        response = self.client.post("/predict", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # Verify prediction structure
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 1)
        self.assertEqual(len(data["predictions"][0]), 2)
        
        # Verify prediction is valid (probabilities should sum close to 1)
        pred = data["predictions"][0]
        self.assertAlmostEqual(sum(pred), 1.0, places=1)
        
        # Ensure prediction matches expected class
        expected_class = 0 if sum(test_sample) > 0 else 1
        predicted_class = 0 if pred[0] > pred[1] else 1
        self.assertEqual(predicted_class, expected_class)

    def test_end_to_end_workflow(self):
        """
        Test the complete workflow from training to prediction.
        
        This test simulates a real-world scenario where a model is trained,
        saved, loaded into the serving API, and then used for predictions.
        """
        # Create a new model
        model = MLP(nin=3, nouts=[5, 2])
        
        # Create trainer
        trainer = Trainer(
            model=model,
            learning_rate=0.05,
            epochs=15,
            batch_size=16,
        )
        
        # Train the model with a different dataset
        np.random.seed(100)
        X_new = np.random.randn(100, 3).tolist()
        y_new = [[1, 0] if x[0] > x[1] + x[2] else [0, 1] for x in X_new]
        
        # Train and save
        trainer.train(X_new, y_new)
        model_path = os.path.join(self.model_dir, "workflow_test_model.pth")
        trainer.save_model(model_path)
        
        # Load model into API
        model_name = load_model_from_path(model_path, "workflow_test_model")
        
        # Create test input
        test_input = [[1.0, 0.2, 0.1]]  # Should be class 0
        
        # Make prediction directly from model
        direct_pred = model(test_input)[0]
        direct_class = 0 if direct_pred[0].data > direct_pred[1].data else 1
        
        # Make prediction via API
        request_data = {"inputs": test_input, "model_name": model_name}
        response = self.client.post("/predict", json=request_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        api_pred = response.json()["predictions"][0]
        api_class = 0 if api_pred[0] > api_pred[1] else 1
        
        # Verify both predictions match
        self.assertEqual(api_class, direct_class)
        
        # Create a test dataset for batch evaluation
        test_dataset = [(x, y) for x, y in zip(X_new[:10], y_new[:10])]
        
        # Evaluate both directly and via API
        correct_direct = 0
        correct_api = 0
        
        for x, y in test_dataset:
            # Direct prediction
            direct_result = model([x])[0]
            direct_class = 0 if direct_result[0].data > direct_result[1].data else 1
            expected_class = 0 if y[0] > y[1] else 1
            if direct_class == expected_class:
                correct_direct += 1
                
            # API prediction
            request_data = {"inputs": [x], "model_name": model_name}
            response = self.client.post("/predict", json=request_data)
            api_result = response.json()["predictions"][0]
            api_class = 0 if api_result[0] > api_result[1] else 1
            if api_class == expected_class:
                correct_api += 1
        
        # Both should have similar accuracy
        self.assertEqual(correct_direct, correct_api)


if __name__ == "__main__":
    unittest.main() 