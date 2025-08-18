"""
Simple local test script to verify model creation and loading
Run this locally before deployment to ensure everything works.
"""

import os
import sys

def test_model_creation():
    """Test creating a fallback model"""
    print("🧪 Testing model creation...")
    
    try:
        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Import and run model creation
        from create_fallback_model import create_fallback_model
        model_path = create_fallback_model()
        
        print(f"✅ Model creation successful: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_model_loading():
    """Test loading the created model"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Import and test model loading
        from src.cnn_predictor import MediscopePredictor
        
        predictor = MediscopePredictor("models/pneumonia_model.h5")
        
        if predictor.model is not None:
            print("✅ Model loading successful")
            
            # Test prediction
            import numpy as np
            dummy_input = np.random.random((1, 150, 150, 1))
            prediction = predictor.model.predict(dummy_input, verbose=0)
            print(f"✅ Test prediction successful: {prediction.shape}")
            return True
        else:
            print("❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting local tests...\n")
    
    # Test model creation
    if not test_model_creation():
        print("\n❌ Model creation test failed")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed")
        return False
    
    print("\n✅ All local tests passed!")
    print("🎉 Ready for deployment!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
