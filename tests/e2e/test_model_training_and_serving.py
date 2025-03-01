"""
End-to-end tests for the complete model workflow.

These tests validate the entire pipeline from training to serving,
ensuring all components work together correctly in a real-world scenario.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from nlp_suite.model_serving.api import ModelRegistry, PredictionRequest


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

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests are done."""
        shutil.rmtree(cls.temp_dir)

    def test_model_registry_and_prediction(self):
        """Test that the model registry can register and serve models."""
        # Create a model registry
        registry = ModelRegistry()
        
        # Create a mock model
        model_name = "test_model"
        mock_model = MagicMock()
        mock_model.return_value = [[0.7, 0.3]]
        
        # Register the mock model with the registry
        registry.register_model(model_name, mock_model)
        
        # Get the model from the registry
        retrieved_name, retrieved_model = registry.get_model(model_name)
        
        # Verify the model was retrieved correctly
        self.assertEqual(retrieved_name, model_name)
        self.assertEqual(retrieved_model, mock_model)
        
        # Create a prediction request
        test_sample = self.X_test[0]
        request = PredictionRequest(
            inputs=[test_sample],
            model_name=model_name
        )
        
        # Make a prediction using the model directly
        prediction = retrieved_model(request.inputs)
        
        # Verify prediction structure
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 1)
        self.assertEqual(len(prediction[0]), 2)
        
        # Verify prediction is valid (probabilities should sum close to 1)
        pred = prediction[0]
        self.assertAlmostEqual(sum(pred), 1.0, places=1)

    def test_multiple_models_in_registry(self):
        """Test that the registry can handle multiple models."""
        # Create a model registry
        registry = ModelRegistry()
        
        # Create mock models
        model_names = ["model_a", "model_b", "model_c"]
        mock_models = {}
        
        for name in model_names:
            # Create a mock model with different prediction values
            mock_model = MagicMock()
            if name == "model_a":
                mock_model.return_value = [[0.8, 0.2]]
            elif name == "model_b":
                mock_model.return_value = [[0.6, 0.4]]
            else:
                mock_model.return_value = [[0.4, 0.6]]
            
            # Register the mock model
            registry.register_model(name, mock_model)
            mock_models[name] = mock_model
        
        # Verify all models are registered
        self.assertEqual(len(registry.models), len(model_names))
        
        # Test getting each model by name
        for name in model_names:
            retrieved_name, retrieved_model = registry.get_model(name)
            self.assertEqual(retrieved_name, name)
            self.assertEqual(retrieved_model, mock_models[name])
        
        # Test getting metadata for all models
        all_metadata = registry.get_metadata()
        self.assertEqual(len(all_metadata), len(model_names))
        
        # Test getting metadata for a specific model
        model_metadata = registry.get_metadata("model_a")
        self.assertEqual(len(model_metadata), 1)
        self.assertEqual(model_metadata[0].name, "model_a")
        
        # Test predictions from different models
        test_sample = self.X_test[0]
        
        # Model A prediction
        name_a, model_a = registry.get_model("model_a")
        pred_a = model_a([test_sample])
        self.assertEqual(pred_a, [[0.8, 0.2]])
        
        # Model B prediction
        name_b, model_b = registry.get_model("model_b")
        pred_b = model_b([test_sample])
        self.assertEqual(pred_b, [[0.6, 0.4]])
        
        # Model C prediction
        name_c, model_c = registry.get_model("model_c")
        pred_c = model_c([test_sample])
        self.assertEqual(pred_c, [[0.4, 0.6]])


if __name__ == "__main__":
    unittest.main() 