"""
Deployment configuration for Mediscope AI
This file contains deployment-specific settings and environment variable configurations.
"""

import os

def configure_deployment_environment():
    """
    Configure environment variables for deployment compatibility.
    This should be called at the start of the application.
    """
    # HuggingFace Configuration
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # TensorFlow Configuration (CPU only)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Reduce TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Disable TensorFlow deprecation warnings
    os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"
    
    print("✅ Deployment environment configured")

if __name__ == "__main__":
    configure_deployment_environment()
