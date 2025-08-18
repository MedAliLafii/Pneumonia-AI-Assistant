"""
Deployment setup script for Mediscope AI
This script prepares the application for deployment by ensuring all models are compatible.
"""

import os
import sys
import shutil

def setup_deployment():
    """Setup the application for deployment"""
    print("🚀 Setting up Mediscope AI for deployment...")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✅ Created models directory")
    
    # Check if original model exists and is compatible
    original_model_path = 'models/pneumonia_model.h5'
    fallback_model_path = 'models/pneumonia_model_fallback.h5'
    
    if os.path.exists(original_model_path):
        print("📁 Original model found, testing compatibility...")
        try:
            # Try to import and test the original model
            from src.cnn_predictor import MediscopePredictor
            predictor = MediscopePredictor(original_model_path)
            
            if predictor.model is not None:
                print("✅ Original model is compatible")
                return True
            else:
                print("⚠️ Original model failed to load, creating fallback...")
                
        except Exception as e:
            print(f"⚠️ Original model compatibility test failed: {e}")
            print("🔄 Creating fallback model...")
    else:
        print("📁 No original model found, creating fallback model...")
    
    # Create fallback model
    try:
        from create_fallback_model import create_fallback_model
        create_fallback_model()
        print("✅ Fallback model created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create fallback model: {e}")
        return False

def main():
    """Main deployment setup function"""
    print("🔧 Mediscope AI Deployment Setup")
    print("=" * 40)
    
    if setup_deployment():
        print("\n✅ Deployment setup completed successfully!")
        print("🎉 Ready for deployment!")
        return True
    else:
        print("\n❌ Deployment setup failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
