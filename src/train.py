import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import os

import json # Still needed for json.dump in the __main__ block for default config creation
import joblib
from sklearn.linear_model import LogisticRegression
import os
# Import from your utils.py: load_config, load_data, and the DEFAULT_CONFIG_PARAMS
from utils import load_config, load_data, DEFAULT_CONFIG_PARAMS

def train_model(config_path="config/config.json", model_output_path="model_train.pkl"):
    """
    Loads the digits dataset, reads hyperparameters from config.json,
    trains a LogisticRegression model, and saves it.
    Args:
        config_path (str): Path to the configuration JSON file.
        model_output_path (str): Path to save the trained model.
    """
    # Use the utility function to load config (which now includes validation)
    config = load_config(config_path)

    # Use the utility function to load data
    X, y = load_data()

    # Extract hyperparameters directly. No defaults here, as load_config ensures their presence and type.
    C = config["C"]
    solver = config["solver"]
    max_iter = config["max_iter"]

    print(f"Training with C={C}, solver={solver}, max_iter={max_iter}")

    # Train model
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_output_path)
    print(f"Model trained and saved to {model_output_path}")

if __name__ == "__main__":
    config_dir = "config"
    config_file_path = os.path.join(config_dir, "config.json")

    os.makedirs(config_dir, exist_ok=True) 

    if not os.path.exists(config_file_path):
        
        with open(config_file_path, "w") as f:
            json.dump(DEFAULT_CONFIG_PARAMS, f, indent=4) 
        print(f"Created default config file at {config_file_path} using DEFAULT_CONFIG_PARAMS from utils.")

   
    train_model(config_path=config_file_path)