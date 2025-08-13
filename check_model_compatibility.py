#!/usr/bin/env python3
"""
Model compatibility checker for the CNN model.
"""

import os
import h5py
import tensorflow as tf
import numpy as np

def check_model_file():
    """Check the model file structure and compatibility."""
    model_path = "models/improved_pneumonia_cnn.h5"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"📁 File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        # Check HDF5 file structure
        with h5py.File(model_path, 'r') as f:
            print("\n📊 HDF5 File Structure:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj).__name__}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
            
            f.visititems(print_structure)
        
        # Try to get model config
        print("\n🔧 Model Configuration:")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            print(f"  Number of layers: {len(model.layers)}")
            print(f"  Model name: {model.name}")
            
            # Check first layer details
            first_layer = model.layers[0]
            print(f"  First layer: {first_layer.__class__.__name__}")
            print(f"  First layer config: {first_layer.get_config()}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")
        return False

def suggest_fixes():
    """Suggest fixes for common compatibility issues."""
    print("\n💡 Suggested Fixes:")
    print("1. Try downgrading TensorFlow: pip install tensorflow==2.12.0")
    print("2. Try upgrading TensorFlow: pip install tensorflow==2.15.0")
    print("3. Re-save the model with current TensorFlow version")
    print("4. Use model conversion tools if available")

def main():
    """Main function."""
    print("🔍 CNN Model Compatibility Checker")
    print("=" * 50)
    
    success = check_model_file()
    
    if not success:
        suggest_fixes()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Model appears to be compatible")
    else:
        print("❌ Model has compatibility issues")

if __name__ == "__main__":
    main()
