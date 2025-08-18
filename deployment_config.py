"""
Deployment configuration for Mediscope AI
This file contains deployment-specific settings and environment variable configurations.
"""

import os
import sys

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
    
    # Additional TensorFlow stability settings
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_GPU_ALLOCATOR"] = "cpu"
    
    # Disable TensorFlow library loading issues
    os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "true"
    
    # Set Python path to avoid import issues
    if os.path.dirname(__file__) not in sys.path:
        sys.path.insert(0, os.path.dirname(__file__))
    
    # Additional deployment settings
    os.environ["PYTHONPATH"] = os.path.dirname(__file__)
    
    print("✅ Deployment environment configured")

def check_tensorflow_availability():
    """
    Check if TensorFlow can be imported safely.
    Returns True if TensorFlow is available, False otherwise.
    """
    try:
        # Configure environment before import
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import tensorflow as tf
        print("✅ TensorFlow is available")
        return True
    except Exception as e:
        print(f"⚠️ TensorFlow is not available: {e}")
        return False

if __name__ == "__main__":
    configure_deployment_environment()
    check_tensorflow_availability()
