# scripts/register_external_model.py
import mlflow
import torch
import yaml
import os
import argparse
import json
import sys
import numpy as np

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the project root (assuming scripts/ or external_models/ is one level down from root)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Append src to sys.path to import model definition
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from model import UNetGenerator

def load_config(default_config_filename="config.yaml"):
    # Construct path to config.yaml relative to the project root
    config_path = os.path.join(PROJECT_ROOT, 'config', default_config_filename)
    print(f"Step 1: Attempting to load config from: {config_path}") # Debug print
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        print(f"SCRIPT_DIR: {SCRIPT_DIR}")
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        raise # Re-raise the exception

def register_kaggle_model(pth_file_path, registered_model_name, config, tracking_uri,  run_name="Register Kaggle Model", kaggle_params_file=None):
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config['MLFLOW_EXPERIMENT_NAME']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print("Step 4", experiment)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    print("step 5")
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID for registering external model: {run_id}")

        # Log the original .pth file as an artifact
        mlflow.log_artifact(pth_file_path, artifact_path="original_kaggle_checkpoint")
        print(f"Logged {pth_file_path} as artifact.")

        # Log any Kaggle training parameters if provided
        if kaggle_params_file and os.path.exists(kaggle_params_file):
            with open(kaggle_params_file, 'r') as f:
                kaggle_params = json.load(f)
            mlflow.log_params({f"kaggle_{k}": v for k, v in kaggle_params.items()})
            mlflow.log_artifact(kaggle_params_file, artifact_path="original_kaggle_checkpoint")


        # Instantiate your model architecture
        # Ensure these params match how gen_AB was defined in Kaggle
        # Or load them from kaggle_params if available
        img_size = config.get('IMG_SIZE', 256) # Or from kaggle_params
        channels_gray = config.get('CHANNELS_IMG_GRAY', 1)
        channels_color = config.get('CHANNELS_IMG_COLOR', 3)

        # Create an instance of the generator
        # The UNetGenerator class must be defined as it was during Kaggle training
        model_instance = UNetGenerator(in_channels=channels_gray, out_channels=channels_color)
        
        # Load the state dict
        # Map location to cpu for broader compatibility when logging,
        # as the inference environment (Streamlit) might be CPU.
        state_dict = torch.load(pth_file_path, map_location=torch.device('cpu'), weights_only=False)
        model_instance.load_state_dict(state_dict)
        model_instance.eval() # Set to evaluation mode
        print("External model loaded into UNetGenerator instance.")

        # Log the model to MLflow, which will register it
        # Create a sample input for signature inference (important!)
        # This should match the expected input shape for your gen_AB
        sample_input_tensor = torch.randn(1, channels_gray, img_size, img_size)

        with torch.no_grad():
            sample_output_tensor = model_instance(sample_input_tensor)

        sample_input_numpy = sample_input_tensor.cpu().numpy()

        try:
            signature = mlflow.models.infer_signature(sample_input_tensor, sample_output_tensor)
            print("Model signature inferred successfully.")
        except Exception as e:
            print(f"Warning: Could not infer model signature: {e}. Proceeding without explicit signature.")
            signature = None


        mlflow.pytorch.log_model(
            pytorch_model=model_instance,
            artifact_path="generator_AB_from_kaggle",
            registered_model_name=registered_model_name,
            input_example=sample_input_numpy, # *** USE NUMPY ARRAY HERE ***
            signature=signature # Pass the inferred signature
        )
        print(f"Kaggle model registered as a new version of '{registered_model_name}' in MLflow.")
        
        mlflow.set_tag("source_training_platform", "kaggle")
        mlflow.set_tag("original_checkpoint_file", os.path.basename(pth_file_path))

    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register an externally trained PyTorch model (.pth) to MLflow.")
    parser.add_argument("--pth-file", required=True, help="Path to the .pth model state_dict file.")
    parser.add_argument("--model-name", help="MLflow registered model name. Defaults to config.")
    parser.add_argument("--kaggle-params", help="Optional JSON file with Kaggle training parameters.", default=None)
    print("Step 1")
    args = parser.parse_args()
    app_config = load_config()
    print("Step 2")
    reg_model_name = args.model_name if args.model_name else app_config.get("MLFLOW_MODEL_NAME_GEN_AB")
    if not reg_model_name:
        raise ValueError("Registered model name not provided via argument or found in config.")
    print("Step3")
    register_kaggle_model(args.pth_file, reg_model_name, app_config, tracking_uri=app_config['MLFLOW_TRACKING_URI'], kaggle_params_file=args.kaggle_params)
