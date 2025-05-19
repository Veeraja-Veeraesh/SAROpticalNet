# Dockerfile.app

# Start with a Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME=/app
ENV GCP_PROJECT_ID=""

WORKDIR ${APP_HOME}

# Install system dependencies if any (e.g., for Pillow or other libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx libglib2.0-0 \ # Example for OpenCV/GUI-less matplotlib if needed
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file for the app
COPY requirements_app.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy the application code (Streamlit app and the src directory for model definitions)
COPY streamlit_app.py .
COPY src/ src/
COPY config/config.yaml config/ 

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Healthcheck (Optional but good for Cloud Run)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#   CMD curl -f http://localhost:8501/_stcore/health || exit 1
# Note: Streamlit's health check endpoint might change or need specific configuration.
# For now, we'll rely on Cloud Run's default TCP health check.

# Command to run the Streamlit application
# We use environment variables for PORT which Cloud Run will provide.
# Also, server.headless=true is important for running in containers.
# server.enableCORS=false might be needed depending on setup, but usually not for simple cases.
# server.runOnSave=false is good practice for containers.
# ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=${PORT}", "--server.headless=true", "--server.runOnSave=false"]
ENTRYPOINT streamlit run streamlit_app.py --server.port=${PORT} --server.headless=true --server.runOnSave=false
