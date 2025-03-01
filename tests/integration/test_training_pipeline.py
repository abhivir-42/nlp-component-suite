"""
Integration tests for the training pipeline.

These tests validate that the model training pipeline works correctly
by testing the interaction between different components.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nlp_suite.architectures.mlp import MLP
from nlp_suite.nn_core.value import Value
from nlp_suite.training_pipelines.trainer import Trainer


class TestTrainingPipeline(unittest.TestCase):
    """Test the complete training pipeline."""

    def setUp(self):
        """Set up test case."""
        # Create temporary directory for saving models
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = Path(self.temp_dir.name)

        # Create a small dataset for testing
        np.random.seed(42)
        self.X = np.random.randn(100, 2).tolist()
        self.y = [[1, 0] if x[0] + x[1] > 0 else [0, 1] for x in self.X]

    def tearDown(self):
        """Tear down test case."""
        self.temp_dir.cleanup()

    def test_train_and_evaluate(self):
        """Test that model training and evaluation work together."""
        # Create a model
        model = MLP(
            nin=2,
            nouts=[4, 2],
            activation=lambda x: x.tanh(),
            activation_final=lambda x: x.sigmoid(),
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            learning_rate=0.1,
            epochs=10,
            batch_size=16,
            checkpoint_dir=self.save_dir,
        )

        # Train the model
        history = trainer.train(self.X, self.y)

        # Check that history contains loss values
        self.assertIn("loss", history)
        self.assertEqual(len(history["loss"]), 10)

        # Check that loss decreased during training
        self.assertLess(history["loss"][-1], history["loss"][0])

        # Check that a checkpoint was saved
        checkpoint_files = list(self.save_dir.glob("*.pth"))
        self.assertGreaterEqual(len(checkpoint_files), 1)

        # Evaluate the model
        metrics = trainer.evaluate(self.X, self.y)

        # Check that metrics contain accuracy
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.5)  # At least better than random

    def test_save_and_load_model(self):
        """Test that model saving and loading work correctly."""
        # Create a model
        model = MLP(
            nin=2,
            nouts=[4, 2],
            activation=lambda x: x.tanh(),
            activation_final=lambda x: x.sigmoid(),
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            learning_rate=0.1,
            epochs=5,
            batch_size=16,
            checkpoint_dir=self.save_dir,
        )

        # Train the model
        trainer.train(self.X, self.y)

        # Save the model
        model_path = self.save_dir / "test_model.pth"
        trainer.save_model(model_path)

        # Check that the model file exists
        self.assertTrue(model_path.exists())

        # Create a new model with the same architecture
        new_model = MLP(
            nin=2,
            nouts=[4, 2],
            activation=lambda x: x.tanh(),
            activation_final=lambda x: x.sigmoid(),
        )

        # Create a new trainer with the loaded model
        new_trainer = Trainer(
            model=new_model,
            learning_rate=0.1,
            epochs=5,
            batch_size=16,
            checkpoint_dir=self.save_dir,
        )

        # Load the model
        new_trainer.load_model(model_path)

        # Make predictions with both models
        # Generate a test input
        test_input = [[0.5, -0.5]]
        
        # Ensure predictions are the same (or very close)
        original_pred = model(test_input)[0]
        loaded_pred = new_model(test_input)[0]
        
        # Compare data from Value objects
        self.assertAlmostEqual(original_pred[0].data, loaded_pred[0].data, places=5)
        self.assertAlmostEqual(original_pred[1].data, loaded_pred[1].data, places=5)


if __name__ == "__main__":
    unittest.main() 