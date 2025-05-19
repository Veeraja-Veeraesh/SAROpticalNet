# streamlit_app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
import yaml
import mlflow

# Assuming model.py (with UNetGenerator) is in src/
# To make it importable, we might need to adjust Python path or package structure slightly.
# For now, let's assume we can import it.
# A common way if streamlit_app.py is in root and src/ is a module:
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import UNetGenerator # Ensure CHANNELS_IMG_GRAY, CHANNELS_IMG_COLOR are accessible or passed

# --- Configuration ---
def load_app_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_app_config()
MLFLOW_TRACKING_URI = config.get('MLFLOW_TRACKING_URI', "http://localhost:5000") # Default if not in config
REGISTERED_MODEL_NAME = config.get('MLFLOW_MODEL_NAME_GEN_AB', "SAR_Generator_AB")
MODEL_ALIAS_TO_LOAD = "prod-candidate" # Or "champion", "latest-validated" - this is key!
DEVICE = "cuda" if torch.cuda.is_available() and config.get('DEVICE_STREAMLIT', 'cpu') == 'cuda' else "cpu"
IMG_SIZE = config.get('IMG_SIZE', 256)
CHANNELS_IMG_GRAY = config.get('CHANNELS_IMG_GRAY', 1)
CHANNELS_IMG_COLOR = config.get('CHANNELS_IMG_COLOR', 3)

st.set_page_config(layout="wide", page_title="SAR Image Colorization")

# --- MLflow Model Loading ---
@st.cache_resource # Cache the loaded model
def load_mlflow_model(model_name, alias):
    try:
        model_uri = f"models:/{model_name}@{alias}"
        st.info(f"Attempting to load model from MLflow: {model_uri} to device: {DEVICE}")
        # Load the PyTorch model (the "pytorch_model.bin" part)
        # mlflow.pytorch.load_model expects the artifact path within the run,
        # but models:/<name>@<alias> points to the registered model version.
        # We need to get the actual run artifact URI for the model files.

        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        try:
            model_version_details = client.get_model_version_by_alias(model_name, alias)
            if not model_version_details:
                st.error(f"No model version found for alias '{alias}' under model '{model_name}'.")
                return None
            source_uri = model_version_details.source # This is like runs:/<run_id>/<artifact_path>
            st.info(f"Found model version {model_version_details.version} with source URI: {source_uri}")
        except mlflow.exceptions.MlflowException as e:
            st.error(f"MLflow API error fetching model by alias: {e}")
            st.warning(f"Ensure MLflow server is running at {MLFLOW_TRACKING_URI} and model '{model_name}@{alias}' exists.")
            return None


        # Now load the actual PyTorch model using the source_uri
        # The model was logged with mlflow.pytorch.log_model, so it should be a PyTorch native model.
        # The logged artifact path was e.g., "generator_AB_model"
        # So, the model files (pytorch_model.bin etc.) are at source_uri + "/pytorch_model.bin"
        
        # mlflow.pytorch.load_model can take the runs:/ URI directly
        loaded_model = mlflow.pytorch.load_model(model_uri=source_uri, map_location=DEVICE)
        loaded_model.to(DEVICE)
        loaded_model.eval()
        st.success(f"Model '{model_name}@{alias}' (version {model_version_details.version}) loaded successfully to {DEVICE}!")
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model '{model_name}@{alias}': {e}")
        st.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        return None

# --- Image Transformations ---
# Input transform for S1 (grayscale)
transform_input_s1 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Denormalize function for displaying output
def denormalize_image(tensor_image):
    # Assumes tensor_image is in [-1, 1] range
    img = tensor_image.cpu().clone().detach().squeeze(0) # Remove batch dim
    img = (img * 0.5) + 0.5  # To [0, 1]
    img = img.permute(1, 2, 0) # C, H, W -> H, W, C
    img = img.numpy()
    img = np.clip(img, 0, 1)
    return img

# --- Streamlit UI ---
st.title("🛰️ Sentinel-1 SAR Image Colorization App ")
st.markdown(f"""
This app uses a CycleGAN-based model to colorize Sentinel-1 (SAR) grayscale images
into pseudo Sentinel-2 (optical) like RGB images.
*Currently loading model: **{REGISTERED_MODEL_NAME}@{MODEL_ALIAS_TO_LOAD}***
""")

# Load the model
generator_model = load_mlflow_model(REGISTERED_MODEL_NAME, MODEL_ALIAS_TO_LOAD)

if generator_model:
    uploaded_file = st.file_uploader("Upload a Sentinel-1 Grayscale Image (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is not None:
        try:
            input_image_pil = Image.open(uploaded_file).convert("L") # Convert to grayscale
            st.image(input_image_pil, caption="Uploaded S1 Image (Grayscale)", width=300)

            # Preprocess the image
            input_tensor = transform_input_s1(input_image_pil).unsqueeze(0).to(DEVICE) # Add batch dimension

            # Perform inference
            with torch.no_grad():
                st.info("Colorizing image... 🎨")
                generated_s2_tensor = generator_model(input_tensor)
                st.info("Colorization complete!")

            # Denormalize and display output
            generated_s2_display = denormalize_image(generated_s2_tensor)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original S1 (Input)")
                # For grayscale display from PIL directly:
                st.image(input_image_pil, use_column_width=True)
            with col2:
                st.subheader("Generated S2-like (Output)")
                st.image(generated_s2_display, caption="Colorized Output", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e) # Shows full traceback in Streamlit for debugging
else:
    st.warning(f"Model '{REGISTERED_MODEL_NAME}@{MODEL_ALIAS_TO_LOAD}' could not be loaded. Please ensure the MLflow server is running and the model alias exists.")

st.sidebar.header("About")
st.sidebar.info(
    "This application demonstrates an MLOps pipeline for SAR image colorization. "
    "It uses a pre-trained generator model fetched from an MLflow Model Registry."
)
st.sidebar.markdown(f"**MLflow Tracking URI:** `{MLFLOW_TRACKING_URI}`")
st.sidebar.markdown(f"**Registered Model Name:** `{REGISTERED_MODEL_NAME}`")
st.sidebar.markdown(f"**Model Alias Loaded:** `{MODEL_ALIAS_TO_LOAD}`")
st.sidebar.markdown(f"**Inference Device:** `{DEVICE}`")
