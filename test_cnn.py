#!/usr/bin/env python3
"""
Test script to verify CNN model loading and basic functionality.
"""

import os
import sys
from src.cnn_predictor import PneumoniaPredictor

def test_cnn_model():
    """Test the CNN model loading and basic functionality."""
    print("🧪 Testing CNN Model...")
    
    # Check if model file exists
    model_path = "models/improved_pneumonia_cnn.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    
    # Try to load the model
    try:
        predictor = PneumoniaPredictor(model_path)
        if predictor.model is None:
            print("❌ Failed to load CNN model")
            return False
        
        print("✅ CNN model loaded successfully")
        
        # Test model summary
        print("\n📊 Model Summary:")
        predictor.model.summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CNN model: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting CNN Model Tests...\n")
    
    success = test_cnn_model()
    
    if success:
        print("\n✅ All tests passed! CNN model is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the model file and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
