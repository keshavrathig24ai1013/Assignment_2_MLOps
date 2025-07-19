# tests/test_train.py
import pytest
import json
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
# Import train_model directly from src.train
from src.train import train_model
# Import load_config and load_data from src.utils, as they are now used by train_model
from src.utils import load_config, load_data

# Fixture for a temporary config file
# pytest's tmp_path_factory provides a unique temporary directory for each test function/fixture
@pytest.fixture(scope="module")
def temp_config_file(tmp_path_factory):
    """
    Creates a temporary config.json file for testing purposes.
    The scope="module" means it's created once for all tests in this module.
    """
    config_dir = tmp_path_factory.mktemp("config_test_dir") # Create a unique temp directory
    config_path = config_dir / "config.json"
    config_data = {
        "C": 0.5,
        "solver": "lbfgs", # Changed to lbfgs as it's a common default and widely supported
        "max_iter": 200
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path

# Fixture for a temporary model output path
@pytest.fixture(scope="module")
def temp_model_path(tmp_path_factory):
    """
    Creates a temporary path for the model output file.
    """
    model_dir = tmp_path_factory.mktemp("models_test_dir") # Create a unique temp directory
    return model_dir / "test_model.pkl"

class TestTrainingPipeline:
    """
    Contains unit tests for the training pipeline components.
    """

    def test_config_file_loading(self, temp_config_file):
        """
        Test that the configuration file loads successfully and contains required hyperparameters
        with correct data types.
        This test directly uses the load_config utility function.
        """
        config = load_config(temp_config_file) # Use the utility function

        assert isinstance(config, dict)
        assert "C" in config
        assert "solver" in config
        assert "max_iter" in config

        assert isinstance(config["C"], (int, float))
        assert isinstance(config["solver"], str)
        assert isinstance(config["max_iter"], int)

    def test_model_creation(self, temp_config_file, temp_model_path):
        """
        Verify that the training function returns a LogisticRegression object
        and the model has been fitted (by checking for attributes like .coef_ or .classes_).
        """
        # train_model now internally uses load_config and load_data
        # We pass the temporary paths to ensure isolation for testing
        train_model(config_path=temp_config_file, model_output_path=temp_model_path)

        # Load the saved model to verify its type and fitted status
        loaded_model = joblib.load(temp_model_path)

        assert isinstance(loaded_model, LogisticRegression)
        # Check if the model has been fitted by verifying the existence of expected attributes
        assert hasattr(loaded_model, 'coef_')
        assert hasattr(loaded_model, 'classes_')

    def test_model_accuracy(self, temp_config_file, temp_model_path):
        """
        Train the model on the digits dataset and evaluate it on the same data.
        Check that the accuracy is above some threshold to verify correctness of training logic.
        """
        # train_model now internally uses load_config and load_data
        # We pass the temporary paths to ensure isolation for testing
        train_model(config_path=temp_config_file, model_output_path=temp_model_path)
        loaded_model = joblib.load(temp_model_path)

        # Load the dataset for evaluation (using the utility function)
        X, y = load_data()

        accuracy = loaded_model.score(X, y)
        print(f"\nModel accuracy in test_model_accuracy: {accuracy}") # Print for visibility in test logs
        assert accuracy > 0.90 
                              