"""
Test script to verify model loading before deployment
This script tests both the CNN model and embeddings to ensure they load correctly.
"""

import os
import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        from PIL import Image
        print(f"✅ Pillow {Image.__version__}")
        
        import streamlit as st
        print(f"✅ Streamlit {st.__version__}")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ LangChain HuggingFace")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_cnn_model():
    """Test CNN model loading"""
    print("\n🔍 Testing CNN model loading...")
    
    try:
        from src.cnn_predictor import MediscopePredictor
        
        # Test with original model
        predictor = MediscopePredictor("models/pneumonia_model.h5")
        
        if predictor.model is not None:
            print("✅ Original CNN model loaded successfully")
            
            # Test prediction
            import numpy as np
            dummy_input = np.random.random((1, 150, 150, 1))
            prediction = predictor.model.predict(dummy_input, verbose=0)
            print(f"✅ Test prediction successful: {prediction.shape}")
            return True
        else:
            print("❌ Original CNN model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ CNN model test failed: {e}")
        traceback.print_exc()
        return False

def test_embeddings():
    """Test embeddings loading"""
    print("\n🔍 Testing embeddings loading...")
    
    try:
        from src.helper import download_hugging_face_embeddings
        
        # Set environment variables
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        embeddings = download_hugging_face_embeddings()
        
        if embeddings is not None:
            print("✅ Embeddings loaded successfully")
            
            # Test embedding
            test_text = "test"
            embedding = embeddings.embed_query(test_text)
            print(f"✅ Test embedding successful: {len(embedding)} dimensions")
            return True
        else:
            print("❌ Embeddings failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting deployment compatibility tests...\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your dependencies.")
        return False
    
    # Test CNN model
    if not test_cnn_model():
        print("\n⚠️ CNN model test failed. Consider creating a fallback model.")
        print("Run: python create_fallback_model.py")
    
    # Test embeddings
    if not test_embeddings():
        print("\n⚠️ Embeddings test failed. The application will use fallback embeddings.")
    
    print("\n✅ All tests completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
