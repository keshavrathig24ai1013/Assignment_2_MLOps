import json
from sklearn.datasets import load_digits
import os

# Define default configuration parameters here for bootstrapping
# This is a fixed set of defaults for creating the config.json file if it doesn't exist
DEFAULT_CONFIG_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000
}

def load_config(config_path="config/config.json"):
    """
    Loads configuration from a JSON file and validates required parameters.
    Args:
        config_path (str): The path to the configuration JSON file.
    Returns:
        dict: The loaded and validated configuration dictionary.
    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file is not valid JSON.
        ValueError: If a required parameter is missing in the config.
        TypeError: If a parameter has an incorrect data type.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validation for required parameters and their expected types
        required_params = {
            "C": (float, int),
            "solver": str,
            "max_iter": int
        }

        for param, expected_type in required_params.items():
            if param not in config:
                raise ValueError(f"Missing required parameter '{param}' in config file: {config_path}")
            if not isinstance(config[param], expected_type):
                raise TypeError(f"Parameter '{param}' in config must be of type {expected_type}, "
                                f"but got {type(config[param]).__name__}.")

        print(f"Configuration loaded and validated from {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from {config_path}: {e.msg}", e.doc, e.pos)
    except Exception as e:
        print(f"Error validating config file {config_path}: {e}")
        raise

def load_data():
    """
    Loads the digits dataset from sklearn.datasets.
    Returns:
        tuple: A tuple containing (features, target) (X, y).
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    print("Digits dataset loaded.")
    return X, y